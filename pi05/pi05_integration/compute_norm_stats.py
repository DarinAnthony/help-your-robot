"""Run normalization-stat computation from the vendored openpi tree."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPENPI = ROOT / "third_party" / "openpi"
sys.path.insert(0, str(OPENPI / "src"))
sys.path.insert(0, str(OPENPI / "packages" / "openpi-client" / "src"))


def main() -> None:
    runpy.run_path(str(OPENPI / "scripts" / "compute_norm_stats.py"), run_name="__main__")


if __name__ == "__main__":
    main()
