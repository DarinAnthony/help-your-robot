#!/bin/bash
# Repository-owned pi0.5 GPU trainer. It uses the integrated source under third_party/openpi;
# The pinned CUDA environment is created inside this checkout by pi05_env.sh.
#
# Example:
#   sbatch --export=ALL,CONFIG=pi05_lowvar_v2_fullft_h32_20k,EXP=my_run \
#     scripts/cluster/pi05_train_gpu.sbatch
# Registered runtime example:
#   sbatch --export=ALL,TRAIN_EXPERIMENT=<experiment>,TRAIN_RUNTIME=pi05_gpu,EXP=<run>,\
#MOLMOSPACES_TRAIN_DATA_DIR=<rlds>,MOLMOSPACES_CHECKPOINT_DIR=<checkpoints>,\
#MOLMOSPACES_NORM_ASSETS_DIR=<assets>,MOLMOSPACES_NORM_ASSET_ID=<asset>,\
#MOLMOSPACES_FSDP_DEVICES=4 scripts/cluster/pi05_train_gpu.sbatch
#
# The registered full-FT runtime is calibrated for one 141 GB H200. A different GPU topology
# should be represented by a separate runtime profile so its FSDP device count is recorded.
#SBATCH --job-name=pi05_fullft
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=224G
#SBATCH --time=3-00:00:00
#SBATCH --output=/iris/u/boai/slurm/%j.out

set -euo pipefail

REPO="${REPO:-/iris/u/boai/embodiment-scaling-manipulation-data}"
PYTHON="${PI05_GPU_PYTHON:-$REPO/.venv/pi05-gpu/bin/python}"
CONFIG="${CONFIG:-pi05_lowvar_v2_fullft_h32_20k}"
TRAIN_EXPERIMENT="${TRAIN_EXPERIMENT:-}"
TRAIN_RUNTIME="${TRAIN_RUNTIME:-}"
TRAIN_COMMAND="${TRAIN_COMMAND:-train}"
EXP="${EXP:?set EXP to a unique experiment name}"

cd "$REPO"

# SC's default AFS home is not guaranteed to expose the IRIS W&B credentials to batch jobs.
# Load the existing private netrc at runtime without copying the secret into this repository or log.
export HOME="${PI05_HOME:-/iris/u/boai}"
if [[ -z "${WANDB_API_KEY:-}" && -r "$HOME/.netrc" ]]; then
  WANDB_API_KEY="$(awk '
    $1 == "machine" { active = ($2 == "api.wandb.ai") }
    active && $1 == "password" { print $2; exit }
  ' "$HOME/.netrc")"
  export WANDB_API_KEY
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "W&B credential missing: set WANDB_API_KEY or provide $HOME/.netrc" >&2
  exit 2
fi
export WANDB_ENTITY="${WANDB_ENTITY:-boai-01-stanford-university}"
export WANDB_MODE="${WANDB_MODE:-online}"
export MOLMOSPACES_EXPECT_ACCELERATOR=gpu
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/iris/u/boai/.cache/jax}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pi05-matplotlib-${SLURM_JOB_ID}}"

echo "repo=$REPO"
echo "git_sha=$(git rev-parse HEAD)"
echo "config=$CONFIG exp=$EXP"
echo "node=$(hostname) job_id=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Normalization is a dataset-build artifact, never a training-launch side effect. Typed
# experiments validate the configured manifest/hash before constructing the trainer; the
# legacy path requires its existing config-scoped asset and never overwrites it here.
NORM_ASSET_ID="${NORM_ASSET_ID:-molmospaces_lowvar_v2}"
NORM_STATS="$REPO/assets/$CONFIG/$NORM_ASSET_ID/norm_stats.json"
if [[ -n "$TRAIN_EXPERIMENT" ]]; then
  echo "validating_registered_normalization_artifact_for=$TRAIN_EXPERIMENT"
else
  if [[ ! -s "$NORM_STATS" ]]; then
    echo "normalization artifact missing: $NORM_STATS" >&2
    echo "build it explicitly before submitting training" >&2
    exit 2
  fi
  echo "reusing_norm_stats=$NORM_STATS"
fi

if [[ -n "$TRAIN_EXPERIMENT" ]]; then
  if [[ -n "$TRAIN_RUNTIME" ]]; then
    "$PYTHON" -m molmo_spaces.training.run "$TRAIN_COMMAND" \
      --experiment "$TRAIN_EXPERIMENT" --runtime "$TRAIN_RUNTIME" \
      --exp-name "$EXP" "$@"
  else
    "$PYTHON" -m molmo_spaces.training.pi_entrypoint train "$TRAIN_EXPERIMENT" \
      --platform gpu --exp-name "$EXP" "$@"
  fi
else
  "$PYTHON" -m pi05_integration.train "$CONFIG" --exp-name "$EXP" "$@"
fi
