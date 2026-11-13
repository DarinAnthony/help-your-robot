"""Build the identified robots as datagen artifacts (model.xml, params.json) for run_pipeline's --variant_dir.

- ``tidybot_xarm7_measured``: the UCSD TidyBot (TidyBot++ base, xArm7, wrist F/T, G2) on its latest measured rig;
  drawer opening, ``--robot mobile_unified_arm``.
- ``droid_panda``: the stock DROID station on the real Panda's working joint range; pick and place.

Both come from xemb's ``task_set.DEPLOYMENT_ROBOTS`` through ``unified_source.generate_unified_manifold_candidate``
(as xemb's real-rig ``ensure_robot`` and tests do). ``model.xml`` stores absolute paths into the checkout that built
it, so rebuild after moving ``$XEMB``.

    source env.sh && $PY scripts/build_robots.py            # -> $MLSPACES_ARTIFACT_ROOT/robots/<name>/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from molmo_spaces.data_generation.collection.online_embodiment import unified_source as us
from molmo_spaces.data_generation.collection.online_embodiment.task_set import DEPLOYMENT_ROBOTS

ROBOTS = ("tidybot_xarm7_measured", "droid_panda")


def build(name: str, out_root: Path, seed: int = 20261017) -> Path:
    source = us.OERUnifiedManifoldSource(**DEPLOYMENT_ROBOTS[name])
    config = SimpleNamespace(unified_manifold_source=source, root_seed=seed, robot_dataset_name="help")
    artifact = us.generate_unified_manifold_candidate(config, 0, None)
    if not artifact.valid:
        raise RuntimeError(f"{name}: {artifact.failure_reason}")
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.xml").write_text(artifact.model_xml)
    (out / "params.json").write_text(artifact.params_json)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--robots", nargs="+", default=list(ROBOTS), choices=ROBOTS)
    p.add_argument("--out", type=Path, default=Path(os.environ["MLSPACES_ARTIFACT_ROOT"]) / "robots")
    a = p.parse_args()
    for name in a.robots:
        out = build(name, a.out)
        params = json.loads((out / "params.json").read_text())
        print(name, "->", out, "| keys:", sorted(params)[:8])


if __name__ == "__main__":
    main()
