#!/usr/bin/env bash
# Install the tracked pi0.5 embodiment-conditioning integration into an openpi checkout.
# Usage: bash openpi_integration/install_embodiment_conditioning.sh [/path/to/openpi]
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
OPENPI=${1:-/iris/u/boai/openpi_gpu}
MODULE_SRC="$REPO/molmo_spaces/policy/learned_policy/embodiment_conditioning.py"
MODULE_DST="$OPENPI/src/openpi/training/embodiment_conditioning.py"
PATCH="$REPO/openpi_integration/embodiment_conditioning.patch"

test -f "$OPENPI/src/openpi/training/config.py" || {
  echo "Not an openpi checkout: $OPENPI" >&2
  exit 2
}

if patch --dry-run -R -p1 -d "$OPENPI" < "$PATCH" >/dev/null 2>&1; then
  echo "openpi embodiment-conditioning patch is already installed"
else
  patch -p1 -d "$OPENPI" < "$PATCH"
fi
install -m 0644 "$MODULE_SRC" "$MODULE_DST"

echo "Installed embodiment conditioning into $OPENPI"
