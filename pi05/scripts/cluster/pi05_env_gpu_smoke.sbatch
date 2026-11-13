#!/bin/bash
# Build and verify the repository-local Pi GPU environment on a real CUDA device.
# This is an environment smoke test, not a training job.
#SBATCH --job-name=pi05_env_smoke
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/iris/u/boai/slurm/%j.out

set -euo pipefail

REPO="${REPO:-/iris/u/boai/embodiment-scaling-manipulation-data-pi-integration}"
cd "$REPO"

scripts/setup/pi05_env.sh gpu

PYTHON="$REPO/.venv/pi05-gpu/bin/python"
"$PYTHON" -m pytest -q \
  tests/test_pi05_training_registry.py \
  third_party/openpi/src/openpi/training/config_molmospaces_test.py \
  third_party/openpi/src/openpi/models/pi0_rtc_test.py

"$PYTHON" -m molmo_spaces.training.main show \
  pi05_lowvar_v2_lora_h32_20k
