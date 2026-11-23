"""Frozen DINOv2 embeddings of every tree node's decision frames, for the V1 image value function (EXPERIMENT_PLAN.md §5).

    $PY scripts/embed_trees.py --trees data/shelf_pnp/20261114_tree --family expert

Reads each tree's ``tree_<seed>.npz`` (``<path or root>/<camera>``, 180x320 RGB), runs DINOv2 ViT-S/14 (timm
``vit_small_patch14_dinov2.lvd142m``, weights from the Hugging Face hub, cached on first use) at 224x392, keeping the
aspect ratio (16x28 patches), and stores per node and camera the CLS token and the mean patch token (2 x 384). Writes
``<trees>/dinov2_vits14.npz``: ``keys`` ("<tree key>|<path>") and ``emb`` (n, cameras, 768) float16, ``cameras``.
Restartable: nodes already in the output are skipped.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpsim.help.trees import load_trees  # noqa: E402

MODEL = "vit_small_patch14_dinov2.lvd142m"
CAMERAS = ("cam_cabinet", "wrist_camera")
SIZE = (224, 392)
OUT_NAME = "dinov2_vits14.npz"


def load_model(device: str):
    import timm
    import torch

    m = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=SIZE)
    cfg = timm.data.resolve_data_config({}, model=m)
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
    return m.eval().to(device), mean, std


def embed(model, mean, std, imgs: np.ndarray, device: str) -> np.ndarray:
    """(n, 180, 320, 3) uint8 -> (n, 768): CLS token, mean patch token."""
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(imgs).to(device).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=SIZE, mode="bicubic", align_corners=False).clamp(0, 1)
    with torch.no_grad():
        tok = model.forward_features((x - mean) / std)          # (n, 1 + registers + patches, 384)
    npre = getattr(model, "num_prefix_tokens", 1)
    return torch.cat([tok[:, 0], tok[:, npre:].mean(1)], 1).float().cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trees", type=Path, required=True)
    p.add_argument("--family", choices=("expert", "pi05"), default="expert")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default="mps")
    a = p.parse_args()

    out = a.trees / OUT_NAME
    done = {}
    if out.exists():
        z = np.load(out)
        done = dict(zip(z["keys"].tolist(), z["emb"]))
    trees = [t for t in load_trees(a.trees, a.family) if Path(t.frames).exists()]
    todo = [(t, path) for t in trees for path in t.nodes if f"{t.key}|{path}" not in done]
    print(f"{len(trees)} trees, {len(done)} nodes done, {len(todo)} to embed", flush=True)
    if not todo:
        return
    model, mean, std = load_model(a.device)
    t0, buf_keys, buf_imgs = time.time(), [], []

    def flush():
        e = embed(model, mean, std, np.stack(buf_imgs), a.device).reshape(-1, len(CAMERAS), 768)
        for k, v in zip(buf_keys, e):
            done[k] = v.astype(np.float16)
        buf_keys.clear()
        buf_imgs.clear()

    last_frames, z = None, None
    for i, (t, path) in enumerate(todo):
        if t.frames != last_frames:
            z, last_frames = np.load(t.frames), t.frames
        buf_keys.append(f"{t.key}|{path}")
        buf_imgs.extend(z[f"{path or 'root'}/{cam}"] for cam in CAMERAS)
        if len(buf_imgs) >= a.batch:
            flush()
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(todo)} nodes, {2 * (i + 1) / (time.time() - t0):.0f} img/s", flush=True)
    if buf_keys:
        flush()
    keys = sorted(done)
    np.savez(out, keys=np.array(keys), emb=np.stack([done[k] for k in keys]), cameras=np.array(CAMERAS))
    print(f"wrote {out}: {len(keys)} nodes in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
