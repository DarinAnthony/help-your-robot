#!/bin/bash
# Wait for one structurally complete local Orbax checkpoint, then mirror its run lineage to GCS.
# This is an artifact-migration helper only; it does not stop, resume, or alter training.
#
# Required: LOCAL_CHECKPOINT_ROOT, GCS_CHECKPOINT_ROOT, CHECKPOINT_STEP.
# Optional: GCLOUD, CLOUDSDK_CONFIG.
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=pi05_ckpt_upload
#SBATCH --output=/iris/u/boai/slurm/%j.out

set -euo pipefail

: "${LOCAL_CHECKPOINT_ROOT:?set the local checkpoint lineage root}"
: "${GCS_CHECKPOINT_ROOT:?set the destination GCS checkpoint lineage root}"
: "${CHECKPOINT_STEP:?set the zero-based checkpoint directory}"

GCLOUD="${GCLOUD:-/iris/u/boai/packages/google-cloud-sdk/bin/gcloud}"
export CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-/iris/u/boai/.config/gcloud}"
step_dir="${LOCAL_CHECKPOINT_ROOT}/${CHECKPOINT_STEP}"

echo "[pi05_ckpt_upload] waiting for complete local checkpoint ${step_dir}"
until [[ -s "${step_dir}/_CHECKPOINT_METADATA" \
      && -s "${step_dir}/params/_METADATA" \
      && -s "${step_dir}/params/manifest.ocdbt" \
      && -s "${step_dir}/train_state/_METADATA" \
      && -s "${step_dir}/train_state/manifest.ocdbt" ]]; do
    sleep 60
done

"${GCLOUD}" storage rsync --recursive "${LOCAL_CHECKPOINT_ROOT}" "${GCS_CHECKPOINT_ROOT}"
"${GCLOUD}" storage objects describe \
    "${GCS_CHECKPOINT_ROOT}/${CHECKPOINT_STEP}/_CHECKPOINT_METADATA" \
    --format='value(name)'
echo "[pi05_ckpt_upload] uploaded checkpoint ${CHECKPOINT_STEP} to ${GCS_CHECKPOINT_ROOT}"
