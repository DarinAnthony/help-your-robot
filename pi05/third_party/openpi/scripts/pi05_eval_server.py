"""Compatibility entrypoint for the repository-owned pi0.5 server."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pi05_integration.server import main  # noqa: E402


if __name__ == "__main__":
    main()
