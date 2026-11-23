"""pi0.5 on the TidyBot drawer task in sim: a policy for xemb's datagen pipeline in place of the kino planner.

``scripts/run_drawer_pi05.sh`` runs xemb's ``run_pipeline.py`` exactly as the drawer collection does
(``scripts/collect_drawer_open.sh``: the same scene, drawer, robot, start pose, cameras, physics, 250 Hz control and
``judge_success``), but each episode's policy is ``Pi05TidybotPolicy``. The source of its action chunks is
``$HELP_PI05_SOURCE``:

- ``ws://host:port`` or ``wss://<name>.trycloudflare.com``: the Colab policy server (``scripts/openpi_run.py serve
  drawer``);
- ``replay:<trajectories h5>:<traj index>``: a recorded episode's own commands, to test the loop without a model
  (the same seed gives the same start, so a recorded success should open the drawer again).

Observations and actions match ``scripts/to_lerobot.py``'s drawer dataset: ``exo_camera_1`` (rendered from the mast
camera by ``helpsim/xemb_hooks.py``) and ``wrist_camera`` resized to 180x320; state = 7 arm joints, gripper, base
(x, y, yaw) in the episode's first base frame; action = 7 arm joint targets, gripper command / 255, base target in the
same frame. The observation at step k gets the command for step k+1, and each chunk's first ``$HELP_PI05_EXEC``
(default 8) commands are executed, one per policy step (~15 Hz).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

H, W = 180, 320


def to_first_frame(pose: np.ndarray, base0: np.ndarray) -> np.ndarray:
    """[..., (x, y, yaw)] world poses in base0's frame (as ``scripts/to_lerobot.in_first_frame``)."""
    pose = np.atleast_2d(np.asarray(pose, float))
    c, s = np.cos(base0[2]), np.sin(base0[2])
    d = pose[:, :2] - base0[:2]
    yaw = (pose[:, 2] - base0[2] + np.pi) % (2 * np.pi) - np.pi
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], yaw], 1)


def to_world(pose: np.ndarray, base0: np.ndarray) -> np.ndarray:
    """The inverse of ``to_first_frame`` for one pose."""
    c, s = np.cos(base0[2]), np.sin(base0[2])
    x, y, yaw = pose
    return np.array([base0[0] + c * x - s * y, base0[1] + s * x + c * y,
                     (base0[2] + yaw + np.pi) % (2 * np.pi) - np.pi])


def resize(img: np.ndarray) -> np.ndarray:
    if img.shape[:2] == (H, W):
        return img
    from PIL import Image

    return np.asarray(Image.fromarray(np.asarray(img, np.uint8)).resize((W, H), Image.BILINEAR))


class ReplaySource:
    """A recorded trajectory's commands as model-format chunks, read at the policy's step."""

    needs_images = False

    def __init__(self, h5: str, traj: int, horizon: int = 32):
        import h5py

        dec = lambda row: json.loads(bytes(row).rstrip(b"\x00").decode())
        with h5py.File(h5) as h:
            t = h[f"traj_{traj}"]
            q = [dec(r) for r in t["obs/agent/qpos"][:]]
            a = [dec(r) for r in t["actions/joint_pos"][:]]
        base = np.array([x["base"] for x in q], float)
        cmd = np.array([x.get("base", q[k]["base"]) for k, x in enumerate(a)], float)
        self.base0 = base[0]
        self.a = np.concatenate([np.array([x["arm"] for x in a]), np.array([[x["gripper"][0] / 255.0] for x in a]),
                                 to_first_frame(cmd, self.base0)], 1).astype(np.float32)
        self.q0 = q[0]
        self.horizon = horizon

    def infer(self, obs: dict, step: int) -> dict:
        idx = np.clip(np.arange(step + 1, step + 1 + self.horizon), 0, len(self.a) - 1)
        return {"actions": self.a[idx]}


class ServerSource:
    needs_images = True

    def __init__(self, url: str):
        from openpi_client import websocket_client_policy

        self.client = websocket_client_policy.WebsocketClientPolicy(url)

    def infer(self, obs: dict, step: int) -> dict:
        return self.client.infer(obs)


_SOURCE = None


def source():
    """One source per process (the server connection is kept across episodes)."""
    global _SOURCE
    spec = os.environ["HELP_PI05_SOURCE"]
    if spec.startswith("replay:"):
        h5, traj = spec[len("replay:"):].rsplit(":", 1)
        return ReplaySource(h5, int(traj))      # fresh per episode: replay is for single-episode tests
    if _SOURCE is None:
        _SOURCE = ServerSource(spec)
    return _SOURCE


def make_policy_class():
    from molmo_spaces.policy.base_policy import BasePolicy

    class Pi05TidybotPolicy(BasePolicy):
        def __init__(self, config, task=None):
            super().__init__(config, task)
            self.exec_steps = int(os.environ.get("HELP_PI05_EXEC", "8"))
            self.prompt = "open the drawer"
            self.log = []
            self.reset()

        def reset(self):
            self.src = source()
            self.base0, self.chunk, self.i, self.step = None, None, 0, 0
            self.infer_s, self.calls = 0.0, 0

        def model_input(self, o: dict) -> dict:
            q = o["qpos"]
            base = np.asarray(q["base"], float)
            state = np.concatenate([np.asarray(q["arm"], float)[:7], [float(q["gripper"][0])],
                                    to_first_frame(base, self.base0)[0]]).astype(np.float32)
            inp = {"observation/state": state, "prompt": self.prompt}
            if self.src.needs_images:
                inp["observation/exterior"] = resize(o["exo_camera_1"])
                inp["observation/wrist"] = resize(o["wrist_camera"])
            return inp

        def get_action(self, observation):
            o = observation[0] if isinstance(observation, (list, tuple)) else observation
            if self.base0 is None:
                self.base0 = np.asarray(o["qpos"]["base"], float).copy()
                if isinstance(self.src, ReplaySource):     # the replayed episode must start where this one does
                    d = max(np.abs(np.asarray(o["qpos"]["arm"]) - self.src.q0["arm"]).max(),
                            np.abs(self.base0 - self.src.base0).max())
                    print(f"[pi05_drawer] replay start differs from the recording by {d:.2e}", flush=True)
            if self.chunk is None or self.i >= self.exec_steps:
                t = time.perf_counter()
                self.chunk = np.asarray(self.src.infer(self.model_input(o), self.step)["actions"])
                self.infer_s += time.perf_counter() - t
                self.calls += 1
                self.i = 0
            a = self.chunk[self.i]
            self.i += 1
            self.step += 1
            return {"arm": np.asarray(a[:7], float), "gripper": np.array([float(np.clip(a[7] * 255.0, 0.0, 255.0))]),
                    "base": to_world(a[8:11], self.base0)}

        def get_info(self) -> dict:
            info = super().get_info()
            info.update({"policy_name": "pi05_help", "policy_source": os.environ.get("HELP_PI05_SOURCE", ""),
                         "policy_exec_steps": self.exec_steps, "policy_infer_calls": self.calls,
                         "policy_infer_s": round(self.infer_s, 2)})
            return info

    return Pi05TidybotPolicy


def install() -> None:
    """Every episode of the datagen pipeline gets a Pi05TidybotPolicy instead of its configured planner."""
    import molmo_spaces.data_generation.pipeline as pipeline

    cls = make_policy_class()

    def setup_policy(exp_config, task, preloaded_policy=None, datagen_profiler=None, episode_seed=None):
        policy = cls(task.config, task)
        task.register_policy(policy)
        return policy

    pipeline.setup_policy = setup_policy
