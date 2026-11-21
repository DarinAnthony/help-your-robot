"""Run openpi's norm stats, training and policy server on the help! configs (helpsim/openpi_help.py).

Run from an upstream openpi checkout's environment (``uv run`` in it), with this repo importable:

    uv run python <help>/scripts/openpi_run.py norm-stats shelf --repo help/shelf_pi05 --root /content/drive/.../pi05
    uv run python <help>/scripts/openpi_run.py train shelf --repo help/shelf_pi05 --root ... --exp run1 --steps 10000
    uv run python <help>/scripts/openpi_run.py serve shelf --repo help/shelf_pi05 --root ... --exp run1 --step 9999

``--root`` holds ``assets/`` (norm stats) and ``checkpoints/``: on Drive, so a disconnected Colab runtime resumes with
``--resume``. The LeRobot dataset is read from ``$HF_LEROBOT_HOME/<repo>``.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build(a):
    from openpi.training import config as C

    from helpsim import openpi_help

    cfg = openpi_help.train_config(a.task, a.repo, steps=a.steps, batch_size=a.batch_size)
    root = Path(a.root)
    cfg = dataclasses.replace(cfg, assets_base_dir=str(root / "assets"), checkpoint_base_dir=str(root / "checkpoints"),
                              exp_name=a.exp, log_interval=a.log_interval, num_workers=a.num_workers,
                              resume=a.resume, overwrite=a.overwrite, save_interval=min(cfg.save_interval, a.steps))
    C._CONFIGS_DICT[cfg.name] = cfg
    return cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["norm-stats", "train", "serve"])
    p.add_argument("task", choices=["shelf", "drawer"])
    p.add_argument("--repo", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--exp", default="norm")
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-frames", type=int, default=None, help="norm-stats: only this many frames (quick tests)")
    p.add_argument("--step", type=int, default=None, help="serve: checkpoint step (default: the last)")
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
    else:
        from openpi.policies import policy_config
        from openpi.serving import websocket_policy_server

        exp_dir = Path(cfg.checkpoint_base_dir) / cfg.name / cfg.exp_name
        steps = sorted(int(d.name) for d in exp_dir.iterdir() if d.name.isdigit())
        step = a.step if a.step is not None else steps[-1]
        policy = policy_config.create_trained_policy(cfg, exp_dir / str(step))
        print(f"serving {cfg.name}/{cfg.exp_name} step {step} on port {a.port}", flush=True)
        websocket_policy_server.WebsocketPolicyServer(policy, host="0.0.0.0", port=a.port,
                                                      metadata={"task": a.task, "step": step}).serve_forever()


if __name__ == "__main__":
    main()
