#!/usr/bin/env bash
# Wait for a committed TPU checkpoint, stop its producer, and start a repository-owned Pi run.
#
# This is intended for a dedicated single-user TPU VM where two JAX processes cannot share libtpu.
# The producer is stopped only after the requested Orbax commit marker is visible. The script never
# sends SIGKILL: if the old process does not exit cleanly, the new experiment is not started.

set -euo pipefail

: "${WAIT_CHECKPOINT_URI:?set the gs:// path to commit_success.txt}"
: "${PREVIOUS_PID:?set the TPU training process PID to retire after the checkpoint commits}"
: "${EXPERIMENT:?set the repository training experiment name}"
: "${EXP_NAME:?set a unique experiment lineage}"

REPO="${MLSPACES_REPO:-$HOME/embodiment-scaling-manipulation-data}"
PYTHON="${MOLMOSPACES_PI05_TPU_PY:-$REPO/.venv/pi05-tpu/bin/python}"
GCLOUD="${GCLOUD:-gcloud}"
LOG="${LOG:-$REPO/logs/${EXP_NAME}.log}"
POLL_SECONDS="${POLL_SECONDS:-60}"
EXIT_TIMEOUT_SECONDS="${EXIT_TIMEOUT_SECONDS:-900}"

mkdir -p "$(dirname "$LOG")"
echo "[pi05_tpu_handoff] waiting for $WAIT_CHECKPOINT_URI"
until "$GCLOUD" storage objects describe "$WAIT_CHECKPOINT_URI" \
    --format='value(name)' >/dev/null 2>&1; do
  sleep "$POLL_SECONDS"
done

echo "[pi05_tpu_handoff] checkpoint committed; stopping pid=$PREVIOUS_PID"
if kill -0 "$PREVIOUS_PID" 2>/dev/null; then
  kill -TERM "$PREVIOUS_PID"
fi

deadline=$((SECONDS + EXIT_TIMEOUT_SECONDS))
while kill -0 "$PREVIOUS_PID" 2>/dev/null; do
  if (( SECONDS >= deadline )); then
    echo "[pi05_tpu_handoff] old process did not exit; refusing to start a second TPU process" >&2
    exit 1
  fi
  sleep 10
done

if [[ -z "${WANDB_API_KEY:-}" && -r "$HOME/.netrc" ]]; then
  WANDB_API_KEY="$(awk '
    $1 == "machine" { active = ($2 == "api.wandb.ai") }
    active && $1 == "password" { print $2; exit }
  ' "$HOME/.netrc")"
  export WANDB_API_KEY
fi
export WANDB_ENTITY="${WANDB_ENTITY:-boai-01-stanford-university}"
export WANDB_MODE="${WANDB_MODE:-online}"
export MOLMOSPACES_EXPECT_ACCELERATOR=tpu

echo "[pi05_tpu_handoff] launching $EXPERIMENT as $EXP_NAME; log=$LOG"
cd "$REPO"
nohup "$PYTHON" -u -m molmo_spaces.training.pi_entrypoint train "$EXPERIMENT" \
  --platform tpu --exp-name "$EXP_NAME" --overwrite >"$LOG" 2>&1 &
echo "[pi05_tpu_handoff] pid=$!"
