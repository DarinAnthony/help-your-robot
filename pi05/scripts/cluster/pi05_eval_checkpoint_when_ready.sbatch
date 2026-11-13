#!/bin/bash
# Wait cheaply on CPU for one committed Pi checkpoint, then submit up to three matched
# registered video evaluations to the GPU partition.
#
#   sbatch --export=ALL,CHECKPOINT_STEP=3999 scripts/cluster/pi05_eval_checkpoint_when_ready.sbatch
#
# Defaults reproduce the corrected H=32 lowvar-v2 run. CHECKPOINT_ROOT,
# FIRST_CONFIG, SECOND_CONFIG, and the accelerator selector
# are explicit overrides so the same staging path can evaluate repository-registered
# multi-embodiment TPU runs without copying this script. NO_RTC_CONFIG and
# RTC_CONFIG remain backward-compatible aliases for historical callers.
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=1-12:00:00
#SBATCH --job-name=wait_pi05_eval
#SBATCH --output=/iris/u/boai/data/evaluation/logs/wait_pi05_eval_%j.out

set -euo pipefail

REPO="${MLSPACES_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-gs://boai-iris-central2/pi05_ckpts/pi05_lowvar_v2_lora_h32_20k/20260713_pi05_lowvar_v2_lora_h32_20k_corrected}"
: "${CHECKPOINT_STEP:?set CHECKPOINT_STEP to 3999, 7999, 11999, 15999, or 19999}"

GCLOUD="${GCLOUD:-/iris/u/boai/google-cloud-sdk/bin/gcloud}"
export CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-/iris/u/boai/.config/gcloud}"
if [[ ! -x "$GCLOUD" ]]; then
    echo "gcloud executable not found: $GCLOUD" >&2
    echo "stage from an authenticated control environment or set GCLOUD explicitly" >&2
    exit 127
fi
COMMIT_MARKER="${CHECKPOINT_ROOT}/${CHECKPOINT_STEP}/commit_success.txt"
CHECKPOINT_LINEAGE="${CHECKPOINT_LINEAGE:-restart_20260713_2239}"
if [[ -n "${LOCAL_CHECKPOINT_ROOT:-}" ]]; then
    echo "[wait_pi05_eval] ignoring LOCAL_CHECKPOINT_ROOT; the GCS URI owns the shared cache path" >&2
fi
FIRST_CONFIG="${FIRST_CONFIG:-${NO_RTC_CONFIG:-molmo_spaces.evaluation.configs.legacy_replay_eval_configs:LowvarPi05VerifyAllClustersNoRTC}}"
SECOND_CONFIG="${SECOND_CONFIG:-${RTC_CONFIG:-molmo_spaces.evaluation.configs.legacy_replay_eval_configs:LowvarPi05VerifyAllClustersRTC}}"
RUN_SECOND_CONFIG="${RUN_SECOND_CONFIG:-1}"
THIRD_CONFIG="${THIRD_CONFIG:-}"
RUN_THIRD_CONFIG="${RUN_THIRD_CONFIG:-0}"
if [[ "$RUN_SECOND_CONFIG" != "0" && "$RUN_SECOND_CONFIG" != "1" ]]; then
    echo "RUN_SECOND_CONFIG must be 0 or 1" >&2
    exit 2
fi
if [[ "$RUN_THIRD_CONFIG" != "0" && "$RUN_THIRD_CONFIG" != "1" ]]; then
    echo "RUN_THIRD_CONFIG must be 0 or 1" >&2
    exit 2
fi
if [[ "$RUN_THIRD_CONFIG" == "1" && -z "$THIRD_CONFIG" ]]; then
    echo "THIRD_CONFIG is required when RUN_THIRD_CONFIG=1" >&2
    exit 2
fi
mkdir -p /iris/u/boai/data/evaluation/logs

echo "[wait_pi05_eval] waiting for ${COMMIT_MARKER} from $(hostname)"
until "$GCLOUD" storage objects describe "$COMMIT_MARKER" --format='value(name)' >/dev/null 2>&1; do
    sleep 120
done
echo "[wait_pi05_eval] checkpoint committed; staging inference params/assets locally"

# IRIS GPU nodes do not have GCP instance credentials. Stage only inference inputs rather than the
# much larger optimizer/train-state tree, then point both comparison cells at the same local copy.
LOCAL_STEP="$("${REPO}/scripts/cluster/stage_pi05_checkpoint_from_gcs.py" \
    --source "${CHECKPOINT_ROOT}/${CHECKPOINT_STEP}" --gcloud "${GCLOUD}" \
    --gcloud-config "${CLOUDSDK_CONFIG}")"
LOCAL_CHECKPOINT_ROOT="${LOCAL_STEP%/*}"
echo "[wait_pi05_eval] local checkpoint ready at ${LOCAL_STEP}; submitting matched cells"

common_export="ALL,MLSPACES_REPO=${REPO},SET=checkpoint_path=${LOCAL_CHECKPOINT_ROOT} checkpoint_step=${CHECKPOINT_STEP} checkpoint_lineage=${CHECKPOINT_LINEAGE}"
eval_partition="${EVAL_PARTITION:-iris}"
submit_args=(--parsable --partition="$eval_partition")
if [[ -n "${EVAL_GPU_TYPE:-}" ]]; then
    submit_args+=(--gres="gpu:${EVAL_GPU_TYPE}:1")
else
    submit_args+=(--constraint="${EVAL_CONSTRAINT:-hopper}")
fi
if [[ -n "${EVAL_NODELIST:-}" ]]; then
    submit_args+=(--nodelist="$EVAL_NODELIST")
fi

first_job=$(sbatch "${submit_args[@]}" \
    --export="$common_export,CONFIG=${FIRST_CONFIG}" \
    "$REPO/scripts/cluster/eval.sbatch")
# Avoid simultaneous reads of two copies of the same multi-GiB checkpoint from the shared
# filesystem. Four concurrent restores previously exceeded the fixed readiness timeout even though
# the checkpoints were valid.
if [[ "$RUN_SECOND_CONFIG" == "1" ]]; then
    second_job=$(sbatch "${submit_args[@]}" \
        --dependency="afterok:${first_job}" \
        --export="$common_export,CONFIG=${SECOND_CONFIG}" \
        "$REPO/scripts/cluster/eval.sbatch")
    last_job="$second_job"
else
    second_job="disabled"
    last_job="$first_job"
fi
if [[ "$RUN_THIRD_CONFIG" == "1" ]]; then
    third_job=$(sbatch "${submit_args[@]}" \
        --dependency="afterok:${last_job}" \
        --export="$common_export,CONFIG=${THIRD_CONFIG}" \
        "$REPO/scripts/cluster/eval.sbatch")
else
    third_job="disabled"
fi
echo "[wait_pi05_eval] checkpoint=${CHECKPOINT_STEP} first_config=${FIRST_CONFIG} first_job=${first_job} second_config=${SECOND_CONFIG} second_job=${second_job} third_config=${THIRD_CONFIG:-disabled} third_job=${third_job}"
