"""Run openpi's norm stats, training and policy server on the help! configs (helpsim/openpi_help.py).

Run from an upstream openpi checkout's environment (``uv run`` in it), with this repo importable:

    uv run python <help>/scripts/openpi_run.py norm-stats shelf --repo help/shelf_pi05 --root /content/drive/.../pi05
    uv run python <help>/scripts/openpi_run.py train shelf --repo help/shelf_pi05 --root ... --exp run1 --steps 10000
    uv run python <help>/scripts/openpi_run.py check shelf --repo help/shelf_pi05 --root ... --exp run1
    uv run python <help>/scripts/openpi_run.py serve shelf --repo help/shelf_pi05 --root ... --exp run1

``--root`` holds ``assets/`` (norm stats) and, unless ``--ckpt-root`` is given, ``checkpoints/``. The LeRobot dataset is
read from ``$HF_LEROBOT_HOME/<repo>``.

``check`` is an open-loop test on the held-out ``<repo>_val`` dataset (``scripts/to_lerobot.py``): from every
``--stride``-th frame it predicts a chunk and compares the first ``--exec`` steps with the recorded expert commands.
"hold" is the error of keeping the arm where it is, for scale. It writes ``open_loop_check.json`` into the checkpoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build(a):
    from openpi.training import config as C

    from helpsim import openpi_help

    cfg = openpi_help.train_config(a.task, a.repo, steps=a.steps, batch_size=a.batch_size)
    root = Path(a.root)
    cfg = dataclasses.replace(cfg, assets_base_dir=str(root / "assets"),
                              checkpoint_base_dir=str(Path(a.ckpt_root) if a.ckpt_root else root / "checkpoints"),
                              exp_name=a.exp, log_interval=a.log_interval, num_workers=a.num_workers,
                              resume=a.resume, overwrite=a.overwrite,
                              save_interval=min(a.save_interval, a.steps), keep_period=a.keep_period or None)
    C._CONFIGS_DICT[cfg.name] = cfg
    return cfg


def checkpoint(cfg, step: int | None) -> Path:
    exp_dir = Path(cfg.checkpoint_base_dir) / cfg.name / cfg.exp_name
    steps = sorted(int(d.name) for d in exp_dir.iterdir() if d.name.isdigit() and (d / "params").is_dir())
    if not steps:
        raise SystemExit(f"no checkpoints in {exp_dir}")
    return exp_dir / str(step if step is not None else steps[-1])


def open_loop_check(cfg, a) -> None:
    import numpy as np
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi.policies import policy_config

    from helpsim import openpi_help

    ckpt = checkpoint(cfg, a.step)
    policy = policy_config.create_trained_policy(cfg, ckpt)
    ds = LeRobotDataset(a.val_repo or f"{a.repo}_val")
    episodes = list(range(ds.num_episodes))[: a.max_episodes]
    cols = ds.hf_dataset.with_format("numpy")
    arm = 7
    errs, holds, grip, times = [], [], [], []
    for ep in episodes:
        lo, hi = int(ds.episode_data_index["from"][ep]), int(ds.episode_data_index["to"][ep])
        actions = np.stack(cols[lo:hi]["actions"])
        for k in range(lo, hi - a.exec, a.stride):
            item = ds[k]
            obs = {
                "observation/exterior": item["observation.images.exterior"].numpy(),
                "observation/wrist": item["observation.images.wrist"].numpy(),
                "observation/state": item["observation.state"].numpy(),
                "prompt": item["task"],
            }
            t0 = time.perf_counter()
            pred = np.asarray(policy.infer(obs)["actions"])[: a.exec]
            times.append(time.perf_counter() - t0)
            true = actions[k - lo: k - lo + a.exec]
            state = obs["observation/state"]
            errs.append(np.abs(pred[:, :arm] - true[:, :arm]).mean(0))
            holds.append(np.abs(state[None, :arm] - true[:, :arm]).mean(0))
            grip.append(np.abs(pred[:, arm] - true[:, arm]).mean())
            if openpi_help.ACTION_DIM[a.task] > 8:
                errs[-1] = np.concatenate([errs[-1], np.abs(pred[:, 8:] - true[:, 8:]).mean(0)])
    errs, holds = np.array(errs), np.array(holds)
    out = {
        "checkpoint": str(ckpt),
        "episodes": len(episodes),
        "samples": len(errs),
        "arm_mae_rad": float(errs[:, :arm].mean()),
        "arm_hold_mae_rad": float(holds.mean()),
        "arm_mae_per_joint": [round(float(x), 4) for x in errs[:, :arm].mean(0)],
        "gripper_mae": float(np.mean(grip)),
        "infer_s_median": float(np.median(times[1:] if len(times) > 1 else times)),
    }
    if errs.shape[1] > arm:
        out["base_mae_xy_yaw"] = [round(float(x), 4) for x in errs[:, arm:].mean(0)]
    print(json.dumps(out, indent=1))
    (ckpt / "open_loop_check.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["norm-stats", "train", "check", "serve"])
    p.add_argument("task", choices=["shelf", "drawer"])
    p.add_argument("--repo", required=True)
    p.add_argument("--root", required=True, help="assets/ (norm stats) and, by default, checkpoints/")
    p.add_argument("--ckpt-root", default=None, help="checkpoints elsewhere (e.g. local disk, synced to Drive)")
    p.add_argument("--exp", default="norm")
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--keep-period", type=int, default=5000, help="also keep every n-th checkpoint (0: only the latest)")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-frames", type=int, default=None, help="norm-stats: only this many frames (quick tests)")
    p.add_argument("--step", type=int, default=None, help="check/serve: checkpoint step (default: the last)")
    p.add_argument("--stride", type=int, default=15, help="check: a prediction every this many frames")
    p.add_argument("--exec", type=int, default=8, help="check: steps of each chunk compared")
    p.add_argument("--max-episodes", type=int, default=None, help="check: at most this many validation episodes")
    p.add_argument("--val-repo", default=None, help="check: the held-out dataset (default <repo>_val)")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    cfg = build(a)
    openpi_scripts = Path(__import__("openpi").__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(openpi_scripts))

    if a.mode == "norm-stats":
        import compute_norm_stats

        compute_norm_stats.main(cfg.name, max_frames=a.max_frames)
    elif a.mode == "train":
        import train

        train.main(cfg)
    elif a.mode == "check":
        open_loop_check(cfg, a)
    else:
        from openpi.policies import policy_config
        from openpi.serving import websocket_policy_server

        ckpt = checkpoint(cfg, a.step)
        policy = policy_config.create_trained_policy(cfg, ckpt)
        print(f"serving {ckpt} on port {a.port}", flush=True)
        websocket_policy_server.WebsocketPolicyServer(policy, host="0.0.0.0", port=a.port,
                                                      metadata={"task": a.task, "step": int(ckpt.name)}).serve_forever()


if __name__ == "__main__":
    main()
