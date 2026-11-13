"""Legacy native-config entrypoint for the vendored OpenPI trainer.

Usage is identical to upstream, for example::

    .venv/pi05-gpu/bin/python -m pi05_integration.train \
        pi05_lowvar_v2_lora_h32 --exp-name=my_run

New experiments should use ``python -m molmo_spaces.training.run train`` after resources are
allocated so the repository scientific config and execution runtime are both recorded.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from pi05_integration.runtime import initialize_distributed, log_runtime

ROOT = Path(__file__).resolve().parents[1]
OPENPI = ROOT / "third_party" / "openpi"
sys.path.insert(0, str(OPENPI / "src"))
sys.path.insert(0, str(OPENPI / "packages" / "openpi-client" / "src"))


def main() -> None:
    # A multi-process GPU run must join its process group before the runtime log queries devices.
    initialize_distributed("gpu")
    log_runtime()
    runpy.run_path(str(OPENPI / "scripts" / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
