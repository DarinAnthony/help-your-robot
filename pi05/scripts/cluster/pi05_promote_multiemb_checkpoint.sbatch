#!/bin/bash
# Promote a completed 27-state Pi checkpoint screen to decisive evaluations when macro SR >= target.
# This job is submitted with an afterok dependency on the screen evaluation.
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --job-name=promote_pi05_eval
#SBATCH --output=/iris/u/boai/artifact/pi_runs/logs/evaluation/promote_pi05_eval_%j.out

set -euo pipefail

REPO="${MLSPACES_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/iris/u/boai/artifact/pi_runs/checkpoints/local/pi05_9emb_n60_idtext_state_fullft_h32_40k/20260714_pi05_9emb_n60_idtext_state_fullft_h32_40k}"
CHECKPOINT_LINEAGE="${CHECKPOINT_LINEAGE:-20260714_idtext_state_fullft}"
SCREEN_CONFIG="${SCREEN_CONFIG:-molmo_spaces.evaluation.configs.legacy_replay_eval_configs:Smooth9EmbPi05N60IDTextStateRTCCheckpointScreen}"
TARGET_MACRO_SR="${TARGET_MACRO_SR:-0.60}"
EVAL_GPU_TYPE="${EVAL_GPU_TYPE:-h100}"
: "${CHECKPOINT_STEP:?set CHECKPOINT_STEP}"

result="$(${MLSPACES_PYTHON:-/iris/u/boai/miniconda3/envs/mlspaces/bin/python} - \
    "${REPO}" "${SCREEN_CONFIG}" "${CHECKPOINT_ROOT}" "${CHECKPOINT_STEP}" \
    "${CHECKPOINT_LINEAGE}" <<'PY'
import sys

repo, config_ref, checkpoint_root, checkpoint_step, checkpoint_lineage = sys.argv[1:]
sys.path.insert(0, repo)
from molmo_spaces.evaluation.run import load_config_reference

config = load_config_reference(config_ref).model_copy(
    update={
        "checkpoint_path": checkpoint_root,
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_lineage": checkpoint_lineage,
    }
)
config.model_post_init(None)
print(f"{config.out_dir}/result.json")
PY
)"
if [[ ! -s "${result}" ]]; then
    echo "[promote_pi05_eval] missing completed screen result: ${result}" >&2
    exit 1
fi

macro_sr=$(/iris/u/boai/miniconda3/envs/mlspaces/bin/python -c \
    'import json, re, sys
x=json.load(open(sys.argv[1])); groups={}
for e in x.get("episodes", []):
 m=re.search(r"(?:^|_)v(\d{3,})(?:_|$)", str(e.get("run_dir", "")))
 key=m.group(1) if m else str(e.get("run_dir", "unknown"))
 groups.setdefault(key, []).append(bool(e.get("success")))
if len(groups) != 9: raise SystemExit(f"expected 9 embodiments, found {sorted(groups)}")
print(sum(sum(v)/len(v) for v in groups.values())/len(groups))' "${result}")

echo "[promote_pi05_eval] step=${CHECKPOINT_STEP} macro_sr=${macro_sr} target=${TARGET_MACRO_SR}"
if ! /iris/u/boai/miniconda3/envs/mlspaces/bin/python -c \
    'import sys; raise SystemExit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)' \
    "${macro_sr}" "${TARGET_MACRO_SR}"; then
    echo "[promote_pi05_eval] below target; no promotion"
    exit 0
fi

set_overrides="checkpoint_path=${CHECKPOINT_ROOT} checkpoint_step=${CHECKPOINT_STEP} checkpoint_lineage=${CHECKPOINT_LINEAGE}"
configs=(
    "${ACCEPT_CONFIG:-molmo_spaces.evaluation.configs.legacy_replay_eval_configs:Smooth9EmbPi05N60IDTextStateRTCAcceptance}"
    "${NO_RTC_CONFIG:-molmo_spaces.evaluation.configs.legacy_replay_eval_configs:Smooth9EmbPi05N60IDTextStateNoRTCCheckpointScreen}"
    "${VIDEO_CONFIG:-molmo_spaces.evaluation.configs.legacy_replay_eval_configs:Smooth9EmbPi05N60IDTextStateRTCWideVideos}"
)
for config in "${configs[@]}"; do
    eval_job=$(sbatch --parsable --partition="${EVAL_PARTITION:-iris-hi}" \
        --gres="gpu:${EVAL_GPU_TYPE}:1" \
        --export="ALL,MLSPACES_REPO=${REPO},CONFIG=${config},SET=${set_overrides}" \
        "${REPO}/scripts/cluster/eval.sbatch")
    echo "[promote_pi05_eval] config=${config} eval_job=${eval_job}"
done
