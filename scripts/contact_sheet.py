"""Frames across time from episode videos, one row per video, as one PNG (a quick look for bugs).

    source env.sh && $PY scripts/contact_sheet.py --out sheet.png --cols 8 \
        "tomato 1001 cabinet=data/shelf_pnp/20261113/train/tomato/ep_01001_cam_cabinet.mp4" ...

Each row is ``label=path``; the frame's time in the video is printed on it.
"""

from __future__ import annotations

import argparse

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def row(label: str, path: str, cols: int, height: int) -> Image.Image:
    reader = imageio.get_reader(path)
    fps, n = reader.get_meta_data().get("fps", 1.0), reader.count_frames()
    picks = set(np.linspace(0, n - 1, cols).round().astype(int).tolist())
    frames = {i: f for i, f in enumerate(reader) if i in picks}  # streamed: only the picked frames are kept
    tiles = []
    for i in sorted(frames):
        im = Image.fromarray(frames[i])
        im = im.resize((round(im.width * height / im.height), height))
        ImageDraw.Draw(im).text((4, height - 14), f"{i / fps:.1f}s", fill=(255, 255, 0))
        tiles.append(im)
    out = Image.new("RGB", (sum(t.width for t in tiles), height + 18), "black")
    ImageDraw.Draw(out).text((4, 3), f"{label}   ({n} frames, {path.rsplit('/', 1)[-1]})", fill="white")
    x = 0
    for t in tiles:
        out.paste(t, (x, 18))
        x += t.width
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("rows", nargs="+", help="label=path.mp4")
    p.add_argument("--out", required=True)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--height", type=int, default=150)
    a = p.parse_args()
    rows = [row(*r.rsplit("=", 1), a.cols, a.height) for r in a.rows]
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "black")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(a.out)
    print(a.out, sheet.size)


if __name__ == "__main__":
    main()
