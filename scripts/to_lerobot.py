"""Successful expert episodes as LeRobot datasets for pi0.5 fine-tuning with upstream openpi (TASK_SPEC.md §2-3).

Two datasets, one per task, each frame:

| key | shelf (DROID) | drawer (TidyBot) |
|---|---|---|
| ``observation.images.exterior`` | ``cam_cabinet`` | the mast camera (``exo_camera_1``) |
| ``observation.images.wrist`` | ``wrist_camera`` | ``wrist_camera`` |
| ``observation.state`` | arm q (7), gripper (1) | arm q (7), gripper (1), base x, y, yaw in the episode's first base frame (3) |
| ``actions`` | arm joint targets (7), gripper command / 255 (1) | the same, then the base target in the first base frame (3) |
| ``task`` | ``put the {object} on the shelf`` | ``open the drawer`` |

Images are stored as 180x320 H.264 video (openpi pads them to 224x224). Actions are absolute; the training config decides
whether the arm becomes deltas. Both sources record a step's state after the physics step together with the command
that drove it, so observation k is paired with the command of step k + 1 and the last observation is dropped.

    ~/micromamba/envs/lerobot/bin/python scripts/to_lerobot.py shelf --out data/lerobot \
        data/shelf_pnp/pi05_demos/rerender_20261114 data/shelf_pnp/pi05_demos/rerender_20261113 data/shelf_pnp/pi05_demos/new_20261121
    ~/micromamba/envs/lerobot/bin/python scripts/to_lerobot.py drawer --out data/lerobot <xemb run dirs ...>

Shelf: every episode in the roots' ``episodes.jsonl`` files with outcome success (and, for re-renders, ``reproduced``)
and a frame on every tick; training objects only. Drawer: every trajectory with task success.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np

H, W = 180, 320
FPS = 15
TRAIN_OBJECTS = {"tomato", "egg", "drumstick", "wooden_ball", "banana", "waffle", "sponge", "duster", "tape"}


def resize(frames: np.ndarray) -> np.ndarray:
    if frames.shape[1:3] == (H, W):
        return frames
    from PIL import Image

    return np.stack([np.asarray(Image.fromarray(f).resize((W, H), Image.BILINEAR)) for f in frames])


def read_video(path: Path) -> np.ndarray:
    import av   # imageio's ffmpeg reader iterates without end in this env (imageio-ffmpeg 0.8)

    with av.open(str(path)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def features(state_dim: int, action_dim: int, state_names, action_names) -> dict:
    img = {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channel"]}
    return {
        "observation.images.exterior": img,
        "observation.images.wrist": dict(img),
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": [state_names]},
        "actions": {"dtype": "float32", "shape": (action_dim,), "names": [action_names]},
    }


# ------------------------------------------------------------------ shelf
SHELF_STATE = [f"arm_q{i}" for i in range(1, 8)] + ["gripper"]
SHELF_ACTION = [f"arm_target{i}" for i in range(1, 8)] + ["gripper_cmd"]


def shelf_episodes(roots: list[Path]):
    for log in sorted(p for r in roots for p in r.rglob("episodes.jsonl")):
        for line in log.read_text().splitlines():
            r = json.loads(line)
            if r["object"] not in TRAIN_OBJECTS or not r["success"] or r.get("render_every") != 1:
                continue
            if r.get("reproduced") is False:
                continue
            yield log.parent, r


def shelf_frames(odir: Path, r: dict):
    z = np.load(odir / f"ep_{r['seed']:05d}.npz")
    n = len(z["t"])
    if len(z["frame_tick"]) != n:
        raise ValueError(f"{odir} seed {r['seed']}: {len(z['frame_tick'])} frames for {n} ticks")
    ext = resize(read_video(odir / f"ep_{r['seed']:05d}_cam_cabinet.mp4"))
    wrist = resize(read_video(odir / f"ep_{r['seed']:05d}_wrist_camera.mp4"))
    if len(ext) != n or len(wrist) != n:
        raise ValueError(f"{odir} seed {r['seed']}: videos {len(ext)}/{len(wrist)} for {n} ticks")
    state = np.concatenate([z["arm_qpos"], z["gripper_pos"][:, None]], 1).astype(np.float32)
    action = np.concatenate([z["arm_target"], z["gripper_ctrl"][:, None] / 255.0], 1).astype(np.float32)
    task = f"put the {r['object'].replace('_', ' ')} on the shelf"
    for k in range(n - 1):
        yield {"observation.images.exterior": ext[k], "observation.images.wrist": wrist[k],
               "observation.state": state[k], "actions": action[k + 1]}, task


# ------------------------------------------------------------------ drawer
DRAWER_STATE = SHELF_STATE + ["base_x", "base_y", "base_yaw"]
DRAWER_ACTION = SHELF_ACTION + ["base_target_x", "base_target_y", "base_target_yaw"]


def in_first_frame(pose: np.ndarray, base0: np.ndarray) -> np.ndarray:
    """[x, y, yaw] world poses expressed in the frame of base0 (yaw wrapped to [-pi, pi))."""
    c, s = np.cos(base0[2]), np.sin(base0[2])
    d = pose[:, :2] - base0[:2]
    yaw = (pose[:, 2] - base0[2] + np.pi) % (2 * np.pi) - np.pi
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], yaw], 1)


def drawer_episodes(runs: list[Path]):
    import h5py

    for run in runs:
        for f in sorted(glob.glob(str(run / "house_0" / "trajectories_*.h5"))):
            batch = Path(f).stem.removeprefix("trajectories_")
            with h5py.File(f) as h:
                for name in sorted(h, key=lambda s: int(s.split("_")[-1])):
                    if bool(h[name]["success"][:].any()):
                        yield Path(f), batch, int(name.split("_")[-1])


def drawer_frames(h5: Path, batch: str, i: int):
    import h5py

    dec = lambda row: json.loads(bytes(row).rstrip(b"\x00").decode())
    with h5py.File(h5) as h:
        t = h[f"traj_{i}"]
        q = [dec(r) for r in t["obs/agent/qpos"][:]]
        a = [dec(r) for r in t["actions/joint_pos"][:]]
    n = len(q)
    base = np.array([x["base"] for x in q], float)
    base_cmd = np.array([x.get("base", q[k]["base"]) for k, x in enumerate(a)], float)
    base0 = base[0].copy()
    state = np.concatenate([np.array([x["arm"] for x in q]), np.array([[x["gripper"][0]] for x in q]),
                            in_first_frame(base, base0)], 1).astype(np.float32)
    action = np.concatenate([np.array([x["arm"] for x in a]), np.array([[x["gripper"][0] / 255.0] for x in a]),
                             in_first_frame(base_cmd, base0)], 1).astype(np.float32)
    d = h5.parent
    ext = resize(read_video(d / f"episode_{i:08d}_exo_camera_1_{batch}.mp4"))
    wrist = resize(read_video(d / f"episode_{i:08d}_wrist_camera_{batch}.mp4"))
    if len(ext) != n or len(wrist) != n:
        raise ValueError(f"{h5} traj_{i}: videos {len(ext)}/{len(wrist)} for {n} steps")
    for k in range(n - 1):
        yield {"observation.images.exterior": ext[k], "observation.images.wrist": wrist[k],
               "observation.state": state[k], "actions": action[k + 1]}, "open the drawer"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["shelf", "drawer"])
    p.add_argument("sources", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("data/lerobot"))
    p.add_argument("--repo-id", default=None, help="default help/<task>_pi05_<n episodes>")
    p.add_argument("--max-episodes", type=int, default=None)
    a = p.parse_args()

    if a.task == "shelf":
        eps = list(shelf_episodes(a.sources))
        frames_of = lambda e: shelf_frames(*e)
        feats, robot = features(8, 8, SHELF_STATE, SHELF_ACTION), "franka_droid"
    else:
        eps = list(drawer_episodes(a.sources))
        frames_of = lambda e: drawer_frames(*e)
        feats, robot = features(11, 11, DRAWER_STATE, DRAWER_ACTION), "tidybot_xarm7"
    eps = eps[: a.max_episodes]
    repo_id = a.repo_id or f"help/{a.task}_pi05_{len(eps)}"
    os.environ["HF_LEROBOT_HOME"] = str(a.out.resolve())
    import functools

    from lerobot.common.datasets import lerobot_dataset
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    # H.264 rather than LeRobot's default AV1: Colab's FFmpeg (behind torchcodec) may not decode AV1.
    lerobot_dataset.encode_video_frames = functools.partial(lerobot_dataset.encode_video_frames, vcodec="h264")

    root = a.out.resolve() / repo_id
    if root.exists():
        shutil.rmtree(root)
    ds = LeRobotDataset.create(repo_id=repo_id, root=root, robot_type=robot, fps=FPS, features=feats,
                               image_writer_threads=4, image_writer_processes=0)
    index = []
    for n, e in enumerate(eps):
        task = None
        for frame, task in frames_of(e):
            ds.add_frame({**frame, "task": task})
        ds.save_episode()
        index.append({"episode_index": n, "source": [str(x) if isinstance(x, Path) else x for x in
                                                     (e if a.task == "drawer" else (e[0], e[1]["seed"]))], "task": task})
        if n % 50 == 0:
            print(f"{n + 1}/{len(eps)} episodes", flush=True)
    (root / "help_sources.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index))
    print(f"{repo_id}: {len(eps)} episodes -> {root}")


if __name__ == "__main__":
    main()
