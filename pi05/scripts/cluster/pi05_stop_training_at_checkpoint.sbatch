#!/bin/bash
# Release accelerator resources once a requested Pi checkpoint is atomically complete.
#
# This is intentionally separate from the evaluation watcher: it changes scheduler state and
# therefore requires an explicit TRAIN_JOB_ID. The trainer can later resume from the complete
# model + optimizer checkpoint without changing its scientific lineage.
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=2-12:00:00
#SBATCH --job-name=stop_pi05_at_ckpt
#SBATCH --output=/iris/u/boai/artifact/pi_runs/logs/evaluation/stop_pi05_at_ckpt_%j.out

set -euo pipefail

: "${CHECKPOINT_DIR:?set the exact checkpoint step directory}"
: "${TRAIN_JOB_ID:?set the Slurm trainer job to stop after checkpoint finalization}"
POLL_SECONDS="${POLL_SECONDS:-60}"

echo "[stop_pi05_at_ckpt] waiting for ${CHECKPOINT_DIR}; trainer=${TRAIN_JOB_ID}"
until [[ -s "${CHECKPOINT_DIR}/_CHECKPOINT_METADATA" \
      && -s "${CHECKPOINT_DIR}/params/_METADATA" \
      && -s "${CHECKPOINT_DIR}/params/manifest.ocdbt" \
      && -s "${CHECKPOINT_DIR}/train_state/_METADATA" \
      && -s "${CHECKPOINT_DIR}/train_state/manifest.ocdbt" ]]; do
    sleep "${POLL_SECONDS}"
done

echo "[stop_pi05_at_ckpt] checkpoint finalized; stopping trainer=${TRAIN_JOB_ID}"
scancel "${TRAIN_JOB_ID}"
