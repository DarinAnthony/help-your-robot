#!/bin/bash
# Repository-local pi0.5 environment bootstrap.
#
# Usage:
#     scripts/setup/pi05_env.sh gpu
#     scripts/setup/pi05_env.sh tpu
#     PI05_INSTALL_ONLY=1 scripts/setup/pi05_env.sh gpu  # prepare off-node; verify later
#
# OpenPI remains pinned to its validated JAX/Flax/Orbax versions. GPU and TPU use the
# same frozen dependency graph and repository source; only the JAX hardware plugin is
# different. Environments are written to .venv/pi05-{gpu,tpu} and never use or modify
# external OpenPI/MonoPi checkouts.
set -euo pipefail

DEVICE="${1:?usage: scripts/setup/pi05_env.sh gpu|tpu}"
if [[ "$DEVICE" != "gpu" && "$DEVICE" != "tpu" ]]; then
  echo "unsupported device: $DEVICE (expected gpu or tpu)" >&2
  exit 2
fi

REPO="${MLSPACES_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
OPENPI="$(cd "$(dirname "$0")/../.." && pwd)/third_party/openpi"  # help!: always this folder's copy
ENV_DIR="${PI05_ENV_DIR:-$REPO/.venv/pi05-$DEVICE}"
if [[ -n "${MOLMOSPACES_UV:-}" ]]; then
  UV="$MOLMOSPACES_UV"
elif command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
elif [[ -x "$(dirname "$REPO")/.local/bin/uv" ]]; then
  UV="$(dirname "$REPO")/.local/bin/uv"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
  UV="$HOME/.local/bin/uv"
else
  echo "uv not found; set MOLMOSPACES_UV to its absolute path" >&2
  exit 2
fi
export UV_PROJECT_ENVIRONMENT="$ENV_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$(dirname "$REPO")/.cache/uv}"
export GIT_LFS_SKIP_SMUDGE=1

echo "[pi05-env] device=$DEVICE env=$ENV_DIR"
"$UV" sync --project "$OPENPI" --python 3.11 --group rlds --frozen
PYTHON="$ENV_DIR/bin/python"

# dlimp declares the package named `tensorflow`, while the RLDS group deliberately
# installs tensorflow-cpu. Both distributions write the same module tree and can leave
# a mixed, unloadable installation. Reinstall the pinned CPU distribution cleanly.
SITE_PACKAGES="$($PYTHON -c 'import site; print(site.getsitepackages()[0])')"
"$UV" pip uninstall --python "$PYTHON" tensorflow tensorflow-cpu >/dev/null 2>&1 || true
rm -rf "$SITE_PACKAGES/tensorflow" "$SITE_PACKAGES/tensorflow.libs" \
  "$SITE_PACKAGES/tensorflow_cpu.libs"
# The frozen graph already provided TensorFlow's dependencies. Avoid letting the old
# TensorFlow constraint downgrade ml-dtypes below the version required by JAX 0.5.3.
"$UV" pip install --python "$PYTHON" --no-deps "tensorflow-cpu==2.15.0"

# pi05_integration.server speaks the octo eval protocol (msgpack-numpy over websocket);
# the OpenPI lock does not provide the standalone msgpack_numpy module.
"$UV" pip install --python "$PYTHON" "msgpack-numpy==0.4.8"

if [[ "$DEVICE" == "tpu" ]]; then
  "$UV" pip uninstall --python "$PYTHON" jax-cuda12-plugin jax-cuda12-pjrt \
    >/dev/null 2>&1 || true
  "$UV" pip install --python "$PYTHON" "jax[tpu]==0.5.3" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
fi

CA_BUNDLE="$("$PYTHON" -c 'import certifi; print(certifi.where())')"
if [[ ! -s "$CA_BUNDLE" ]]; then
  echo "certifi CA bundle is missing or empty: $CA_BUNDLE" >&2
  exit 2
fi
echo "[pi05-env] CA bundle=$CA_BUNDLE"

if [[ "${PI05_INSTALL_ONLY:-0}" == "1" ]]; then
  echo "[pi05-env] installed without accelerator verification: $PYTHON"
  exit 0
fi

EXPECTED="$DEVICE" "$PYTHON" - <<'PY'
import os

import jax
import tensorflow as tf
import tensorflow_datasets  # noqa: F401
import dlimp  # noqa: F401

expected = os.environ["EXPECTED"]
platforms = sorted({device.platform for device in jax.devices()})
if jax.__version__ != "0.5.3":
    raise RuntimeError(f"expected JAX 0.5.3, found {jax.__version__}")
if expected not in platforms:
    raise RuntimeError(f"expected platform {expected}, found {platforms}")
print({"jax": jax.__version__, "platforms": platforms, "tensorflow_cpu": tf.__version__})
PY

echo "[pi05-env] ready: $PYTHON"
