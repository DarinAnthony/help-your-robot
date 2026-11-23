"""pi0.5 on the shelf task in sim: closed-loop rollouts, attempt boundaries, help decisions and evaluation trees.

The sim (``helpsim.tasks.shelf_pnp.ShelfPnP``) runs here; the policy is anything with openpi's ``infer(obs) ->
{"actions": (32, 8)}``: openpi's websocket client for the Colab policy server (``connect``) or ``ReplayPolicy``, which
plays a recorded expert episode back through the same loop to test it without a model.

Observations and actions match ``scripts/to_lerobot.py``: ``cam_cabinet`` and the wrist camera at 180x320 (openpi pads
them to 224x224), state = 7 arm joints + gripper opening, action = 7 arm joint targets + gripper command / 255. The
observation at tick k gets the command for tick k+1. Every chunk's first ``exec_steps`` commands are executed at 15 Hz.

**Attempt boundaries** (TASK_SPEC.md §4), from sim state on every tick (``AttemptMonitor``):

- ``success``: the object at rest for 0.5 s on the shelf within 5 cm of the target, the gripper open and 10 cm clear
  of it (or 3 s after letting go), so an object the retreat knocks or that rolls away is not counted;
- ``grasp_miss``: after the gripper closes, the object is not lifted 3 cm with both pads on it within 2 s, or the
  gripper opens before that;
- ``slip``: the object leaves the closed gripper while carried and raised and comes to rest on the shelf off the
  target (a weak grip that loses the object during the lift, before it is 3 cm up, is a ``grasp_miss``);
- ``place_miss``: released (or set down) while carried and at rest on the shelf more than 5 cm from the target;
- ``off_shelf`` (terminal): the object left the shelf;
- ``timeout``: none of these within 300 ticks (20 s).

**Decisions** (§5) come before each attempt: ``continue`` (pi0.5 goes on from the current state), ``reset`` (gripper
open, back out of the cabinet, arm home, then pi0.5) or ``ask``. ``run_tree`` runs both continue and reset from every
non-terminal decision point (the first attempt starts at home, where the two are the same): at most 1 + 2 + 4 = 7
attempts per start for 3 attempts.
"""

from __future__ import annotations

import time

import numpy as np

from helpsim.envs.kitchen_station import HOME
from helpsim.tasks.shelf_pnp import CONTROL_HZ, PLACE_TOL_M, SETTLE_S, ShelfPnP

EXEC_STEPS = 8
ATTEMPT_TICKS = 300           # 20 s
REST_S, REST_SPEED = 0.5, 0.02  # at rest: object COM slower than 2 cm/s for 0.5 s
LIFT_M, GRASP_WINDOW_S = 0.03, 2.0
CLEAR_M, CLEAR_WAIT_S = 0.10, 3.0  # a placement is judged once the gripper is 10 cm away (or 3 s after letting go)
CLOSING, HOLDING, OPEN = 0.5, 0.9, 0.5   # gripper command thresholds (fraction of 255)
TERMINAL = ("success", "off_shelf")


class AttemptMonitor:
    """Watches one attempt; ``update()`` after every tick returns the outcome once the attempt has ended."""

    def __init__(self, env: ShelfPnP, z_rest: float):
        self.env = env
        self.z_rest = z_rest     # the object's COM height resting on the shelf (after reset)
        self.ticks = 0
        self.prev = env.obj_com()
        self.rest = 0
        cmd = self.cmd()
        # An attempt that starts with the object held (a continue after a timeout) is already carrying it.
        self.carried = cmd >= HOLDING and env.pads_touching() == 2 and self.prev[2] > z_rest + LIFT_M
        self.was_closing = cmd >= CLOSING
        self.close_tick = None
        self.z_close = self.tcp_z_close = None
        self.release_tick = None
        self.lost = False

    def cmd(self) -> float:
        return float(self.env.data.ctrl[self.env.grip_act]) / 255.0

    @property
    def released(self) -> bool:
        return self.release_tick is not None

    def target_err(self, p) -> float:
        return float(np.linalg.norm(p[:2] - self.env.target[:2]))

    def update(self) -> str | None:
        env = self.env
        self.ticks += 1
        p = env.obj_com()
        speed = float(np.linalg.norm(p - self.prev)) * CONTROL_HZ
        self.prev = p
        self.rest = self.rest + 1 if speed < REST_SPEED else 0
        at_rest = self.rest >= round(REST_S * CONTROL_HZ)
        cmd, pads = self.cmd(), env.pads_touching()
        tcp = env.data.site_xpos[env.site]
        clear = float(np.linalg.norm(tcp - p)) > CLEAR_M

        if not env.on_shelf() and (at_rest or p[2] < env.shelf_top - 0.10):
            return "off_shelf"
        closing = cmd >= CLOSING
        if closing and not self.was_closing and not self.carried:
            self.close_tick, self.z_close, self.tcp_z_close = self.ticks, p[2], float(tcp[2])
        self.was_closing = closing

        if self.carried:
            if not self.released and not self.lost:
                if cmd < HOLDING:
                    self.release_tick = self.ticks
                elif pads < 2:
                    # Contact lost with the gripper lowered back to the shelf is a placement; raised, a slip.
                    if self.tcp_z_close is not None and tcp[2] > self.tcp_z_close + 0.015:
                        self.lost = True
                    else:
                        self.release_tick = self.ticks
            if self.lost and at_rest:
                return "success" if self.target_err(p) < PLACE_TOL_M else "slip"
            waited = self.released and self.ticks - self.release_tick >= CLEAR_WAIT_S * CONTROL_HZ
            if self.released and at_rest and (cmd < OPEN or pads == 0) and (clear or waited):
                return "success" if self.target_err(p) < PLACE_TOL_M else "place_miss"
        elif self.close_tick is not None:
            if p[2] >= self.z_close + LIFT_M and pads == 2:
                self.carried = True
            elif not closing or self.ticks - self.close_tick >= GRASP_WINDOW_S * CONTROL_HZ:
                return "grasp_miss"
        elif at_rest and pads == 0 and cmd < OPEN and clear and self.target_err(p) < PLACE_TOL_M:
            return "success"     # pushed there
        if self.ticks >= ATTEMPT_TICKS:
            return "timeout"
        return None


class ShelfRollout:
    def __init__(self, obj: str, exec_steps: int = EXEC_STEPS, render: bool = True, **faults):
        # Frames only when asked for (observations, decision points): render_every never fires.
        self.env = ShelfPnP(obj, render=render, render_every=10 ** 9, **faults)
        self.obj = obj
        self.exec_steps = exec_steps
        self.prompt = f"put the {obj.replace('_', ' ')} on the shelf"

    # ------------------------------------------------------------ policy I/O
    def state(self) -> np.ndarray:
        env = self.env
        return np.concatenate([env.arm_q(), [env.data.actuator_length[env.grip_act]]]).astype(np.float32)

    def obs(self, images: bool = True) -> dict:
        o = {"observation/state": self.state(), "prompt": self.prompt}
        if images:
            f = self.env.decision_frames()
            o["observation/exterior"], o["observation/wrist"] = f["cam_cabinet"], f["wrist_camera"]
        return o

    def apply(self, action: np.ndarray) -> None:
        d, env = self.env.data, self.env
        d.ctrl[env.arm_act] = action[:7]
        d.ctrl[env.grip_act] = float(np.clip(action[7] * 255.0, 0.0, 255.0))

    def infer(self, policy) -> np.ndarray:
        return np.asarray(policy.infer(self.obs(getattr(policy, "needs_images", True)))["actions"])

    # ------------------------------------------------------------ attempts
    def attempt(self, policy) -> tuple[str, dict]:
        env = self.env
        mon = AttemptMonitor(env, self.z_rest)
        t0, tick0, calls, infer_s = time.perf_counter(), len(env.rec.rows["t"]), 0, 0.0
        while True:
            t = time.perf_counter()
            chunk = self.infer(policy)
            infer_s += time.perf_counter() - t
            calls += 1
            for a in chunk[: self.exec_steps]:
                self.apply(a)
                env._tick("policy")
                outcome = mon.update()
                if outcome:
                    return outcome, {"ticks": mon.ticks, "tick_start": tick0, "carried": mon.carried,
                                     "grasped": mon.close_tick is not None, "infer_calls": calls,
                                     "infer_s": round(infer_s, 2), "wall_s": round(time.perf_counter() - t0, 2)}

    def reset_arm(self) -> None:
        """Reset-and-retry: open the gripper, back 14 cm out of the cabinet (where reachable), joint-space to home."""
        env, d = self.env, self.env.data
        env._gripper(False, "reset")
        env._move(d.site_xpos[env.site] - 0.14 * env.into, d.site_xmat[env.site].reshape(3, 3).copy(), "reset")
        cur, home = d.ctrl[env.arm_act].copy(), np.array(HOME)
        n = max(1, int(np.ceil(np.abs(home - cur).max() / 0.06)))
        for i in range(1, n + 1):
            d.ctrl[env.arm_act] = cur + (home - cur) * i / n
            env._tick("reset")
        env._hold(SETTLE_S, "reset")

    def spread(self, policy, n: int) -> dict:
        """pi0.5's uncertainty at this state: the spread of n sampled chunks from the same observation."""
        o = self.obs()
        c = np.stack([np.asarray(policy.infer(o)["actions"]) for _ in range(n)])
        e = self.exec_steps
        return {"n": n, "arm_std_exec": float(c[:, :e, :7].std(0).mean()), "arm_std_chunk": float(c[:, :, :7].std(0).mean()),
                "grip_std_exec": float(c[:, :e, 7].std(0).mean()),
                "arm_end_std": [round(float(x), 4) for x in c[:, e - 1, :7].std(0)]}

    def decision(self, policy=None, n_samples: int = 0) -> dict:
        dp = {"state": self.env.decision_state()}
        if n_samples and policy is not None:
            dp["spread"] = self.spread(policy, n_samples)
        return dp

    # ------------------------------------------------------------ episodes and trees
    def run_episode(self, policy, rng: np.random.Generator, decide=None, max_attempts: int = 3,
                    n_samples: int = 0) -> dict:
        """One episode. ``decide(k, previous_outcomes, decision_point) -> "continue" | "reset" | "ask"`` before each
        attempt (default: continue). Stops at success, off_shelf, ask or ``max_attempts``."""
        self.env.reset(rng)
        self.z_rest = float(self.env.obj_com()[2])
        attempts, decisions = [], []
        for k in range(max_attempts):
            dp = self.decision(policy, n_samples)
            choice = decide(k, [a["outcome"] for a in attempts], dp) if decide else "continue"
            if choice == "reset" and k == 0:
                choice = "continue"     # the first attempt starts at home: a reset would change nothing
            decisions.append({"attempt": k, "choice": choice, **dp})
            if choice == "ask":
                break
            if choice == "reset":
                self.reset_arm()
            outcome, info = self.attempt(policy)
            attempts.append({"attempt": k, "outcome": outcome, **info})
            if outcome in TERMINAL:
                break
        last = attempts[-1]["outcome"] if attempts else None
        return {"success": last == "success", "asked": decisions[-1]["choice"] == "ask",
                "final_outcome": "asked" if decisions[-1]["choice"] == "ask" else last,
                "attempts": attempts, "decisions": decisions,
                "start_obj_pos": self.env.start_obj[0].tolist(), "target": self.env.target.tolist()}

    def run_tree(self, policy, seed: int, max_attempts: int = 3, n_samples: int = 0) -> tuple[list[dict], dict]:
        """From the start of ``seed``: every continue (C) / reset (R) sequence of up to ``max_attempts`` attempts, each
        branch from a snapshot of its parent's end state. pi0.5 samples, so each branch is one draw. Returns the nodes
        (root first; ``path`` e.g. "CR": the first attempt, then a reset and a second; ``outcome`` of the attempt that
        ended at the node) and each node's decision frames by path."""
        self.env.reset(np.random.default_rng(seed))
        self.z_rest = float(self.env.obj_com()[2])
        nodes, frames = [], {}

        def visit(path: str, outcome: str | None, info: dict) -> None:
            leaf = outcome in TERMINAL or len(path) >= max_attempts
            nodes.append({"path": path, "depth": len(path), "outcome": outcome, **info,
                          **self.decision(policy, 0 if leaf else n_samples)})
            frames[path] = self.env.decision_frames()
            if leaf:
                return
            start = self.env.snapshot()
            for act in ("C",) if not path else ("C", "R"):
                self.env.restore(start)
                if act == "R":
                    self.reset_arm()
                out, inf = self.attempt(policy)
                visit(path + act, out, inf)

        visit("", None, {})
        return nodes, frames


class ReplayPolicy:
    """Plays a recorded expert episode (``ep_<seed>.npz``) back as action chunks, read at the sim's current tick, so the
    rollout loop and the attempt monitor can be tested against the expert's labelled attempts without a model."""

    needs_images = False

    def __init__(self, npz, env: ShelfPnP, horizon: int = 32):
        z = np.load(npz)
        self.a = np.concatenate([z["arm_target"], z["gripper_ctrl"][:, None] / 255.0], 1).astype(np.float32)
        self.env, self.horizon = env, horizon

    def infer(self, obs: dict) -> dict:
        t = len(self.env.rec.rows["t"]) - 1
        return {"actions": self.a[np.clip(np.arange(t + 1, t + 1 + self.horizon), 0, len(self.a) - 1)]}


def connect(url: str):
    """openpi's websocket client for the policy server, e.g. ``wss://<name>.trycloudflare.com`` or ``ws://host:8000``."""
    from openpi_client import websocket_client_policy

    return websocket_client_policy.WebsocketClientPolicy(url)
