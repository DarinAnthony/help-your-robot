"""Print the task/scene/dataset/embodiment composition of a pi0.5 RLDS config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPENPI = ROOT / "third_party" / "openpi"
sys.path.insert(0, str(OPENPI / "src"))
sys.path.insert(0, str(OPENPI / "packages" / "openpi-client" / "src"))

from openpi.training import config as _config  # noqa: E402
from openpi.training.dataset_mixture import format_markdown, scan_rlds_mixture  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi05_lowvar_v2_lora_h32")
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()
    train_config = _config.get_config(args.config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    print(format_markdown(scan_rlds_mixture(data_config, max_episodes=args.max_episodes)))


if __name__ == "__main__":
    main()
