#!/bin/bash
# Summarize a completed R1--R5 latest-checkpoint video sweep.
# RESULT_CELLS is a semicolon-separated LABEL=/path/to/result.json list.
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=pi05_rstudy_collect
#SBATCH --output=/iris/u/boai/data/evaluation/logs/pi05_rstudy_collect_%j.out

set -euo pipefail

REPO="${MLSPACES_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
: "${RESULT_CELLS:?set semicolon-separated LABEL=/path/to/result.json entries}"
: "${SUMMARY_PATH:?set SUMMARY_PATH}"

cell_args=()
missing=()
IFS=';' read -r -a cells <<< "${RESULT_CELLS}"
for cell in "${cells[@]}"; do
    label="${cell%%=*}"
    result="${cell#*=}"
    if [[ ! -s "${result}" ]]; then
        missing+=("${label}=${result}")
        continue
    fi
    cell_args+=(--cell "${cell}")
done

mkdir -p "$(dirname "${SUMMARY_PATH}")"
{
    if (( ${#cell_args[@]} )); then
        "${REPO}/.venv/pi05-gpu/bin/python" \
            "${REPO}/scripts/analysis/summarize_multiemb_eval.py" \
            "${cell_args[@]}" --target 0.60
    else
        echo "# Balanced multi-embodiment evaluation"
        echo
        echo "No evaluation cell completed successfully."
    fi
    if (( ${#missing[@]} )); then
        echo
        echo "## Missing or failed cells"
        echo
        for cell in "${missing[@]}"; do
            echo "- \`${cell}\`"
        done
    fi
} | tee "${SUMMARY_PATH}"

echo "[pi05_rstudy_collect] summary=${SUMMARY_PATH}"
