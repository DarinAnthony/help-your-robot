#!/bin/bash
# Snapshot the newest committed R1--R5 checkpoints and launch matched saved-video evaluations.
# Submit with --begin=now+7hours to answer the seven-hour progress question without selecting an
# in-flight checkpoint. GCS checkpoints are mirrored locally before GPU inference because SC's JAX
# runtime does not have application-default GCS credentials.
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=pi05_rstudy_latest
#SBATCH --output=/iris/u/boai/data/evaluation/logs/pi05_rstudy_latest_%j.out

set -euo pipefail

REPO="${MLSPACES_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
GCLOUD="${GCLOUD:-/iris/u/boai/packages/google-cloud-sdk/bin/gcloud}"
EVAL_PARTITION="${EVAL_PARTITION:-iris}"
EVAL_GPU_TYPE="${EVAL_GPU_TYPE:-a6000}"
EVAL_TIME="${EVAL_TIME:-06:00:00}"
EVAL_CPUS="${EVAL_CPUS:-16}"
EVAL_MEM="${EVAL_MEM:-128G}"
STAMP="${SLURM_JOB_ID}"
MANIFEST="/iris/u/boai/data/evaluation/logs/pi05_rstudy_latest_${STAMP}.tsv"
SUMMARY="/iris/u/boai/data/evaluations/20260716_pi05_n60_r1_r5_latest_${STAMP}.md"

export HOME=/iris/u/boai
mkdir -p /iris/u/boai/data/evaluation/logs

latest_local_step() {
    local root="$1" latest=-1 dir step
    for dir in "${root}"/[0-9]*; do
        [[ -d "${dir}" ]] || continue
        step="${dir##*/}"
        [[ "${step}" =~ ^[0-9]+$ ]] || continue
        if [[ -s "${dir}/commit_success.txt" ]] || {
            [[ -s "${dir}/_CHECKPOINT_METADATA" ]] \
                && [[ -s "${dir}/params/_METADATA" ]] \
                && [[ -s "${dir}/params/manifest.ocdbt" ]];
        }; then
            (( step > latest )) && latest="${step}"
        fi
    done
    (( latest >= 0 )) || return 1
    printf '%s\n' "${latest}"
}

latest_gcs_step() {
    local root="$1" objects uri parent step latest=-1
    if ! objects=$("${GCLOUD}" storage ls "${root}/*/commit_success.txt" 2>/dev/null); then
        objects=$("${GCLOUD}" storage ls "${root}/*/_CHECKPOINT_METADATA")
    fi
    while IFS= read -r uri; do
        [[ -n "${uri}" ]] || continue
        parent="${uri%/*}"
        step="${parent##*/}"
        [[ "${step}" =~ ^[0-9]+$ ]] || continue
        (( step > latest )) && latest="${step}"
    done <<< "${objects}"
    (( latest >= 0 )) || return 1
    printf '%s\n' "${latest}"
}

materialize_root() {
    local label="$1" root="$2" step dest
    if [[ "${root}" == gs://* ]]; then
        step=$(latest_gcs_step "${root}")
        echo "[pi05_rstudy_latest] resolving ${label} step ${step} in the shared cache" >&2
        dest="$("${REPO}/scripts/cluster/stage_pi05_checkpoint_from_gcs.py" \
            --source "${root}/${step}" --gcloud "${GCLOUD}")"
        printf '%s|%s\n' "${dest%/*}" "${step}"
    else
        step=$(latest_local_step "${root}")
        printf '%s|%s\n' "${root}" "${step}"
    fi
}

labels=(R1 R2 R3 R4 R5)
configs=(
    20260716_pi05_n60_r1_exactid_lora_h32_no_rtc_k16_videos
    20260716_pi05_n60_r2_exactid_fullft_h32_no_rtc_k16_videos
    20260716_pi05_n60_r3_v013_v066_exactid_lora_h32_no_rtc_k16_videos
    20260716_pi05_n60_r4_v013_exactid_lora_h32_no_rtc_k16_videos
    20260716_pi05_n60_r5_v013_v066_noexplicitid_lora_h32_no_rtc_k16_videos
)
lineages=(
    20260716_r1_exactid_lora_h32_7h
    20260716_r2_exactid_fullft_h32_7h
    20260716_r3_v013_v066_exactid_lora_h32_7h
    20260716_r4_v013_exactid_lora_h32_7h
    20260716_r5_v013_v066_noexplicitid_lora_h32_7h
)
roots=(
    gs://boai-iris-central2/pi05_ckpts/pi05_9emb_n60_currentplanner_phasebalanced_perembnorm_id_state_lora_h32_tpu40k/20260716_r1_exactid_lora_h32_b256_lr4e4_cosine30k_v4_32_normfix
    /iris/u/boai/pi05_ckpts_local/pi05_n60_r2_exactid_fullft_h32_b256_lr4e4_30k/20260716_r2_exactid_fullft_h32_b256_lr4e4_cosine30k_4xh200_normidfix
    gs://boai-iris-central2/pi05_ckpts/pi05_n60_r3_v013_v066_exactid_lora_h32_b64_lr1e4_20k/20260716_r3_v013_v066_exactid_lora_h32_b64_lr1e4_cosine20k_v4_16_offline
    /iris/u/boai/pi05_ckpts_local/pi05_n60_r4_v013_exactid_lora_h32_b64_lr1e4_20k/20260716_r4_v013_exactid_lora_h32_b64_lr1e4_cosine20k_2xh100_normidfix
    gs://boai-iris-central2/pi05_ckpts/pi05_n60_r5_v013_v066_perembnorm_noexplicitid_lora_h32_b64_lr1e4_20k/20260716_r5_v013_v066_perembnorm_noexplicitid_lora_h32_b64_lr1e4_cosine20k_v4_8
)

job_ids=()
result_cells=()
printf 'cell\tcheckpoint_root\tcheckpoint_step\tconfig\teval_job\tresult\n' > "${MANIFEST}"
for index in "${!labels[@]}"; do
    label="${labels[index]}"
    config="${configs[index]}"
    lineage="${lineages[index]}"
    resolved=$(materialize_root "${label}" "${roots[index]}")
    checkpoint_root="${resolved%|*}"
    checkpoint_step="${resolved##*|}"
    set_overrides="checkpoint_path=${checkpoint_root} checkpoint_step=${checkpoint_step} checkpoint_lineage=${lineage}"
    eval_job=$(sbatch --parsable --partition="${EVAL_PARTITION}" \
        --gres="gpu:${EVAL_GPU_TYPE}:1" \
        --time="${EVAL_TIME}" --cpus-per-task="${EVAL_CPUS}" --mem="${EVAL_MEM}" \
        --export="ALL,MLSPACES_REPO=${REPO},CONFIG=${config},SET=${set_overrides}" \
        "${REPO}/scripts/cluster/eval.sbatch")
    result="/iris/u/boai/data/evaluations/${config}/${lineage}/step${checkpoint_step}/result.json"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${label}" "${checkpoint_root}" "${checkpoint_step}" "${config}" "${eval_job}" "${result}" \
        | tee -a "${MANIFEST}"
    job_ids+=("${eval_job}")
    result_cells+=("${label}=${result}")
done

dependency="afterany"
for job_id in "${job_ids[@]}"; do
    dependency="${dependency}:${job_id}"
done
joined_cells=$(IFS=';'; echo "${result_cells[*]}")
collector=$(sbatch --parsable --dependency="${dependency}" \
    --export="ALL,MLSPACES_REPO=${REPO},RESULT_CELLS=${joined_cells},SUMMARY_PATH=${SUMMARY}" \
    "${REPO}/scripts/cluster/pi05_collect_recipe_study_eval.sbatch")

echo "[pi05_rstudy_latest] manifest=${MANIFEST}"
echo "[pi05_rstudy_latest] collector=${collector} summary=${SUMMARY}"
