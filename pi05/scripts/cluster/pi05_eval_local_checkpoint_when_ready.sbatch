#!/bin/bash
# Wait on a repository-local Pi checkpoint and submit one standardized evaluation.
#
# Example (nine-embodiment 8k screen):
#   sbatch --export=ALL,CHECKPOINT_STEP=3999 \
#     scripts/cluster/pi05_eval_local_checkpoint_when_ready.sbatch
# Or one process can watch a complete schedule:
#   sbatch --export=ALL,CHECKPOINT_STEPS='3999 7999 11999' \
#     scripts/cluster/pi05_eval_local_checkpoint_when_ready.sbatch
#
# The watcher itself is a tiny CPU job. The evaluation is submitted only after Orbax has
# finalized the checkpoint root and parameter metadata, so an asynchronous save cannot be read
# halfway through. Pi/JAX inference and MuJoCo EGL rendering are both accelerator-sensitive in
# closed loop, so checkpoint comparisons default to one concrete GPU class (H100). Override with
# EVAL_GPU_TYPE only for an explicitly labelled hardware ablation.
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=2-12:00:00
#SBATCH --job-name=wait_pi05_local_eval
#SBATCH --output=/iris/u/boai/artifact/pi_runs/logs/evaluation/wait_pi05_local_eval_%j.out

set -euo pipefail

REPO="${MLSPACES_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/iris/u/boai/artifact/pi_runs/checkpoints/local/pi05_9emb_n60_idtext_state_fullft_h32_40k/20260714_pi05_9emb_n60_idtext_state_fullft_h32_40k}"
CONFIG="${CONFIG:-molmo_spaces.evaluation.configs.legacy_replay_eval_configs:Smooth9EmbPi05N60IDTextStateRTCCheckpointScreen}"
CHECKPOINT_LINEAGE="${CHECKPOINT_LINEAGE:-20260714_idtext_state_fullft}"
: "${CHECKPOINT_STEPS:=${CHECKPOINT_STEP:-}}"
: "${CHECKPOINT_STEPS:?set CHECKPOINT_STEP or a space-separated CHECKPOINT_STEPS schedule}"
poll_seconds="${POLL_SECONDS:-120}"
if ! [[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "POLL_SECONDS must be a positive integer, got ${poll_seconds}" >&2
    exit 2
fi

eval_gpu_type="${EVAL_GPU_TYPE:-h100}"
submit_args=(
    --parsable
    --partition="${EVAL_PARTITION:-iris-hi}"
    --gres="${EVAL_GRES:-gpu:${eval_gpu_type}:1}"
)
if [[ -n "${EVAL_ACCOUNT:-}" ]]; then
    submit_args+=(--account="${EVAL_ACCOUNT}")
fi
if [[ -n "${EVAL_CONSTRAINT:-}" ]]; then
    submit_args+=(--constraint="${EVAL_CONSTRAINT}")
fi
if [[ -n "${EVAL_NODELIST:-}" ]]; then
    submit_args+=(--nodelist="${EVAL_NODELIST}")
fi
if [[ -n "${EVAL_EXCLUDE:-}" ]]; then
    submit_args+=(--exclude="${EVAL_EXCLUDE}")
fi
if [[ -n "${EVAL_CPUS:-}" ]]; then
    submit_args+=(--cpus-per-task="${EVAL_CPUS}")
fi
if [[ -n "${EVAL_MEM:-}" ]]; then
    submit_args+=(--mem="${EVAL_MEM}")
fi
if [[ -n "${EVAL_TIME:-}" ]]; then
    submit_args+=(--time="${EVAL_TIME}")
fi

mkdir -p /iris/u/boai/artifact/pi_runs/logs/evaluation
for checkpoint_step in ${CHECKPOINT_STEPS}; do
    step_dir="${CHECKPOINT_ROOT}/${checkpoint_step}"
    echo "[wait_pi05_local_eval] waiting for ${step_dir} from $(hostname)"
    until [[ -s "${step_dir}/_CHECKPOINT_METADATA" \
          && -s "${step_dir}/params/_METADATA" \
          && -s "${step_dir}/params/manifest.ocdbt" ]]; do
        sleep "$poll_seconds"
    done
    echo "[wait_pi05_local_eval] checkpoint finalized; submitting ${CONFIG}"

    set_overrides="checkpoint_path=${CHECKPOINT_ROOT} checkpoint_step=${checkpoint_step} checkpoint_lineage=${CHECKPOINT_LINEAGE}"
    if [[ -n "${EXTRA_SET:-}" ]]; then
        set_overrides="${set_overrides} ${EXTRA_SET}"
    fi

    eval_job=$(sbatch "${submit_args[@]}" \
        --export="ALL,MLSPACES_REPO=${REPO},CONFIG=${CONFIG},SET=${set_overrides}" \
        "${REPO}/scripts/cluster/eval.sbatch")
    echo "[wait_pi05_local_eval] checkpoint=${checkpoint_step} partition=${EVAL_PARTITION:-iris-hi} account=${EVAL_ACCOUNT:-default} gpu=${eval_gpu_type} constraint=${EVAL_CONSTRAINT:-none} eval_job=${eval_job}"
    if [[ "${AUTO_PROMOTE:-1}" == "1" ]]; then
        promote_job=$(sbatch --parsable --partition=iris --dependency="afterok:${eval_job}" \
            --export="ALL,MLSPACES_REPO=${REPO},CHECKPOINT_ROOT=${CHECKPOINT_ROOT},CHECKPOINT_STEP=${checkpoint_step},CHECKPOINT_LINEAGE=${CHECKPOINT_LINEAGE},SCREEN_CONFIG=${CONFIG},EVAL_GPU_TYPE=${eval_gpu_type},TARGET_MACRO_SR=${TARGET_MACRO_SR:-0.60},ACCEPT_CONFIG=${ACCEPT_CONFIG:-molmo_spaces.evaluation.configs.legacy_replay_eval_configs:Smooth9EmbPi05N60IDTextStateRTCAcceptance},NO_RTC_CONFIG=${NO_RTC_CONFIG:-molmo_spaces.evaluation.configs.legacy_replay_eval_configs:Smooth9EmbPi05N60IDTextStateNoRTCCheckpointScreen},VIDEO_CONFIG=${VIDEO_CONFIG:-molmo_spaces.evaluation.configs.legacy_replay_eval_configs:Smooth9EmbPi05N60IDTextStateRTCWideVideos}" \
            "${REPO}/scripts/cluster/pi05_promote_multiemb_checkpoint.sbatch")
        echo "[wait_pi05_local_eval] checkpoint=${checkpoint_step} promote_job=${promote_job}"
    fi
done
