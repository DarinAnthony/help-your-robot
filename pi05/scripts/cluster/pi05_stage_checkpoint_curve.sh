#!/usr/bin/env bash
# Atomically stage a configured curve of committed Pi checkpoints from GCS.
set -euo pipefail

: "${CAMPAIGN_ID:?set CAMPAIGN_ID}"
: "${EXPERIMENT_NAME:?set EXPERIMENT_NAME}"
: "${RUN_NAME:?set RUN_NAME}"
: "${CHECKPOINT_STEPS:?set space-separated zero-based checkpoint steps}"

gcloud="${GCLOUD_BIN:-/iris/u/boai/packages/google-cloud-sdk/bin/gcloud}"
gcloud_config="${CLOUDSDK_CONFIG:-/scr/boai/.config/gcloud}"
gcs_root="${GCS_CHECKPOINT_ROOT:-gs://boai-iris-central2/pi05_ckpts/${EXPERIMENT_NAME}/${RUN_NAME}}"
stage_helper="${STAGE_HELPER:-/iris/u/boai/bin/boai-stage-gcs-checkpoint}"
staged_root="$("${stage_helper}" --source "${gcs_root}" --print-cache-path)"
state_root="${CAMPAIGN_STATE_ROOT:-/iris/u/boai/artifact/pi_runs/logs/campaigns/${CAMPAIGN_ID}}"
poll_seconds="${POLL_SECONDS:-300}"
if [[ -n "${STAGED_CHECKPOINT_ROOT:-}" ]]; then
    echo "ignoring STAGED_CHECKPOINT_ROOT; the GCS URI owns the shared cache path" >&2
fi

if ! [[ "${poll_seconds}" =~ ^[1-9][0-9]*$ ]]; then
    echo "POLL_SECONDS must be a positive integer, got ${poll_seconds}" >&2
    exit 2
fi
read -r -a steps <<<"${CHECKPOINT_STEPS}"
if [[ "${#steps[@]}" == 0 ]]; then
    echo "CHECKPOINT_STEPS must contain at least one step" >&2
    exit 2
fi
for step in "${steps[@]}"; do
    if ! [[ "${step}" =~ ^[0-9]+$ ]]; then
        echo "checkpoint steps must be nonnegative integers, got ${step}" >&2
        exit 2
    fi
done

if [[ "${PRINT_CONTRACT:-0}" == "1" ]]; then
    printf 'campaign=%s\nexperiment=%s\nrun=%s\nsteps=%s\n' \
        "${CAMPAIGN_ID}" "${EXPERIMENT_NAME}" "${RUN_NAME}" "${CHECKPOINT_STEPS}"
    printf 'gcs_root=%s\nstaged_root=%s\nstate_root=%s\nstage_helper=%s\n' \
        "${gcs_root}" "${staged_root}" "${state_root}" "${stage_helper}"
    exit 0
fi

mkdir -p "${state_root}"
exec 9>"${state_root}/${EXPERIMENT_NAME}.checkpoint-curve-stager.lock"
if ! flock -n 9; then
    echo "another checkpoint-curve stager owns ${EXPERIMENT_NAME}/${RUN_NAME}" >&2
    exit 0
fi

log_event() {
    local step="$1"
    local event_type="$2"
    local detail="$3"
    printf '{"campaign":"%s","checkpoint_step":%s,"event_type":"%s","observed_at":"%s","detail":"%s"}\n' \
        "${CAMPAIGN_ID}" "${step}" "${event_type}" "$(date --iso-8601=seconds)" \
        "${detail}" >>"${state_root}/events.jsonl"
}

checkpoint_complete() {
    local root="$1"
    CLOUDSDK_CONFIG="${gcloud_config}" "${gcloud}" storage ls \
        "${root}/_CHECKPOINT_METADATA" \
        "${root}/params/_METADATA" \
        "${root}/params/manifest.ocdbt" \
        "${root}/assets/commit_success.txt" \
        "${root}/commit_success.txt" >/dev/null 2>&1
}

for step in "${steps[@]}"; do
    source_step="${gcs_root}/${step}"
    while ! checkpoint_complete "${source_step}"; do
        printf '%s checkpoint %s is not committed; waiting %s seconds\n' \
            "$(date --iso-8601=seconds)" "${step}" "${poll_seconds}"
        sleep "${poll_seconds}"
    done

    destination="$("${stage_helper}" --source "${source_step}" \
        --profile pi05_inference --gcloud "${gcloud}" \
        --gcloud-config "${gcloud_config}")"
    log_event "${step}" checkpoint_cache_ready \
        "verified shared inference checkpoint at ${destination}"
done

log_event "${steps[${#steps[@]}-1]}" checkpoint_curve_staged \
    "all configured checkpoint steps are locally complete"
