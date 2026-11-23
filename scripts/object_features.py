"""Physical features of the kitchen objects for the help model (helpsim/help): extents from the compiled collision
geoms (MJCF, body frame), mass and friction from xemb's object registry. Writes helpsim/help/object_features.json.

    source env.sh && $PY scripts/object_features.py
"""

import json
import os
from pathlib import Path

import mujoco
import numpy as np

ODIR = Path(os.environ["XEMB"]) / "envs" / "src_kitchen" / "objects"
OUT = Path(__file__).resolve().parents[1] / "helpsim" / "help" / "object_features.json"


def extents(name: str) -> np.ndarray:
    spec = mujoco.MjSpec.from_file(str(ODIR / f"{name}.xml"))
    for asset in list(spec.meshes) + list(spec.textures):
        if asset.file and not Path(asset.file).is_absolute():
            asset.file = str(ODIR / asset.file)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    body = m.body(name).id
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != body or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        corners = c + h * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        R = d.geom_xmat[g].reshape(3, 3)
        pts.append(corners @ R.T + d.geom_xpos[g] - d.xpos[body])
    pts = np.concatenate(pts)
    return pts.max(0) - pts.min(0)


def main() -> None:
    reg = {o["name"]: o for o in json.loads((ODIR / "registry.json").read_text())["objects"]}
    out = {}
    for name in ("tomato", "egg", "drumstick", "wooden_ball", "banana", "waffle", "sponge", "duster", "tape", "orange",
                 "metal_spoon"):
        e = extents(name)
        long, short = sorted(e[:2], reverse=True)
        out[name] = {"long_m": round(float(long), 4), "short_m": round(float(short), 4), "height_m": round(float(e[2]), 4),
                     "mass_kg": reg[name]["physics"]["mass_kg"], "friction": reg[name]["physics"]["friction"]}
        print(name, out[name])
    OUT.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
