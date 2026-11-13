#!/bin/bash
# Wait for finalized Pi checkpoints in GCS, stage inference assets locally, and submit one
# registered evaluation per checkpoint. Scientific policy/controller/state choices remain in CONFIG.
#
# Required: CHECKPOINT_ROOT, CONFIG, CHECKPOINT_LINEAGE, and either
# CHECKPOINT_STEP or a space-separated CHECKPOINT_STEPS schedule.
# Optional: EVAL_PARTITION, EVAL_GPU_TYPE, EVAL_GPU_PREFERENCE, EVAL_CONSTRAINT,
# EXTRA_SET, POLL_SECONDS. EVAL_GPU_PREFERENCE is a colon-separated ordered list
# (for example h100:h200:a40:titanrtx) resolved from live Slurm allocation data
# only after the checkpoint is committed.
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=2-12:00:00
#SBATCH --job-name=wait_pi05_gcs_eval
#SBATCH --output=/iris/u/boai/artifact/pi_runs/logs/evaluation/wait_pi05_gcs_eval_%j.out

set -euo pipefail

REPO="${MLSPACES_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
: "${CHECKPOINT_ROOT:?set the GCS checkpoint lineage root}"
: "${CONFIG:?set an evaluation module:object reference}"
: "${CHECKPOINT_LINEAGE:?set the output lineage label}"
: "${CHECKPOINT_STEPS:=${CHECKPOINT_STEP:-}}"
: "${CHECKPOINT_STEPS:?set CHECKPOINT_STEP or a space-separated CHECKPOINT_STEPS schedule}"

GCLOUD="${GCLOUD:-/iris/u/boai/packages/google-cloud-sdk/bin/gcloud}"
export CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-/iris/u/boai/.config/gcloud}"
poll_seconds="${POLL_SECONDS:-60}"
if ! [[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "POLL_SECONDS must be a positive integer, got ${poll_seconds}" >&2
    exit 2
fi
mkdir -p /iris/u/boai/artifact/pi_runs/logs/evaluation
if [[ -n "${LOCAL_CHECKPOINT_ROOT:-}" ]]; then
    echo "[wait_pi05_gcs_eval] ignoring LOCAL_CHECKPOINT_ROOT; the GCS URI owns the shared cache path" >&2
fi

choose_available_gpu_type() {
    EVAL_GPU_PREFERENCE="${EVAL_GPU_PREFERENCE}" \
    EVAL_PARTITION="${EVAL_PARTITION:-iris}" \
    python3 - <<'PY'
import json
import os
import re
import subprocess

preferences = [item for item in os.environ["EVAL_GPU_PREFERENCE"].split(":") if item]
partition = os.environ["EVAL_PARTITION"]
nodes = json.loads(subprocess.check_output(["scontrol", "show", "nodes", "--json"]))["nodes"]
free = {gpu_type: 0 for gpu_type in preferences}
unavailable_states = {"DOWN", "DRAINED", "DRAINING", "FAIL", "FAILING", "FUTURE", "MAINTENANCE"}
for node in nodes:
    if partition not in node.get("partitions", []):
        continue
    if unavailable_states.intersection(node.get("state", [])):
        continue
    configured = {
        gpu_type: int(count)
        for gpu_type, count in re.findall(r"gpu:([^:,]+):(\d+)", node.get("gres", ""))
    }
    used = {
        gpu_type: int(count)
        for gpu_type, count in re.findall(r"gpu:([^:,]+):(\d+)", node.get("gres_used", ""))
    }
    for gpu_type in preferences:
        free[gpu_type] += max(0, configured.get(gpu_type, 0) - used.get(gpu_type, 0))

for gpu_type in preferences:
    if free[gpu_type] > 0:
        print(gpu_type)
        break
PY
}

for checkpoint_step in ${CHECKPOINT_STEPS}; do
    if ! [[ "$checkpoint_step" =~ ^[0-9]+$ ]]; then
        echo "checkpoint steps must be non-negative integers, got ${checkpoint_step}" >&2
        exit 2
    fi
    source_step="${CHECKPOINT_ROOT}/${checkpoint_step}"
    marker="${source_step}/commit_success.txt"
    local_step="$("${REPO}/scripts/cluster/stage_pi05_checkpoint_from_gcs.py" \
        --source "${source_step}" --print-cache-path)"
    local_checkpoint_root="${local_step%/*}"
    submission_marker="${local_step}/.eval_submitted_${CONFIG}"

    if [[ -s "$submission_marker" ]]; then
        echo "[wait_pi05_gcs_eval] already submitted config=${CONFIG} checkpoint=${checkpoint_step}: $(<"$submission_marker")"
        continue
    fi
    echo "[wait_pi05_gcs_eval] waiting for ${marker}"
    until "$GCLOUD" storage objects describe "$marker" --format='value(name)' >/dev/null 2>&1; do
        sleep "$poll_seconds"
    done

    local_step="$("${REPO}/scripts/cluster/stage_pi05_checkpoint_from_gcs.py" \
        --source "${source_step}" --gcloud "${GCLOUD}" \
        --gcloud-config "${CLOUDSDK_CONFIG}")"

    set_overrides="checkpoint_path=${local_checkpoint_root} checkpoint_step=${checkpoint_step} checkpoint_lineage=${CHECKPOINT_LINEAGE}"
    if [[ -n "${EXTRA_SET:-}" ]]; then
        set_overrides="${set_overrides} ${EXTRA_SET}"
    fi

    submit_args=(--parsable --partition="${EVAL_PARTITION:-iris}")
    if [[ -n "${EVAL_GPU_TYPE:-}" ]]; then
        selected_gpu_type="${EVAL_GPU_TYPE}"
        submit_args+=(--gres="gpu:${selected_gpu_type}:1")
    elif [[ -n "${EVAL_GPU_PREFERENCE:-}" ]]; then
        selected_gpu_type="$(choose_available_gpu_type)"
        if [[ -z "$selected_gpu_type" ]]; then
            selected_gpu_type="${EVAL_GPU_PREFERENCE%%:*}"
            echo "[wait_pi05_gcs_eval] no preferred GPU is immediately free; queuing on first preference ${selected_gpu_type}"
        fi
        submit_args+=(--gres="gpu:${selected_gpu_type}:1")
    else
        selected_gpu_type="constraint:${EVAL_CONSTRAINT:-hopper}"
        submit_args+=(--constraint="${EVAL_CONSTRAINT:-hopper}")
    fi

    eval_job=$(sbatch "${submit_args[@]}" \
        --export="ALL,MLSPACES_REPO=${REPO},CONFIG=${CONFIG},SET=${set_overrides}" \
        "${REPO}/scripts/cluster/eval.sbatch")
    printf '%s\n' "$eval_job" > "$submission_marker"
    echo "[wait_pi05_gcs_eval] config=${CONFIG} checkpoint=${checkpoint_step} eval_job=${eval_job} gpu=${selected_gpu_type} local=${local_step}"
done
