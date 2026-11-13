#!/usr/bin/env bash
# Start one command detached on this TPU worker; mirror its log to GCS every few minutes and at exit.
#
# Run it on every worker through scripts/cluster/pi05_tpu_multihost.sh from the run's source tree.
# It refuses to start when the attempt's log already exists or when this user already runs a
# molmo_spaces.training process or any process holds the TPU devices. It never kills anything.
#
#   RUN_ID=<run> ATTEMPT=<n> MOLMOSPACES_TPU_LOG_ROOT=gs://<bucket>/<prefix> \
#     scripts/cluster/pi05_tpu_detached.sh <command> [args...]

set -euo pipefail

: "${RUN_ID:?set a unique run identity}"
: "${ATTEMPT:?set the launch attempt number}"
: "${MOLMOSPACES_TPU_LOG_ROOT:?set a same-region gs:// runtime-log root}"
SYNC_SECONDS="${SYNC_SECONDS:-300}"
if [[ "$#" -eq 0 ]]; then
  echo "usage: RUN_ID=... ATTEMPT=... MOLMOSPACES_TPU_LOG_ROOT=gs://... $0 <command> [args...]" >&2
  exit 2
fi

WORKER_ID="$(curl -fsS -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number)"
LOG_DIR="/var/tmp/boai-tpu/${RUN_ID}/logs"
LOG="${LOG_DIR}/attempt-${ATTEMPT}-worker-${WORKER_ID}.log"
URI="${MOLMOSPACES_TPU_LOG_ROOT%/}/attempt-${ATTEMPT}/worker-${WORKER_ID}.log"
mkdir -p "$LOG_DIR"
if [[ -e "$LOG" ]]; then
  echo "refusing to reuse attempt ${ATTEMPT} on worker ${WORKER_ID}: $LOG exists" >&2
  exit 1
fi
if pgrep -u "$(id -u)" -f '^[^ ]*python[^ ]* .*molmo_spaces[.]training[.]run' >/dev/null \
    || [[ "$(sudo -n lsof -w /dev/accel* /dev/vfio/* 2>/dev/null | wc -l)" != 0 ]]; then
  echo "refusing to start: a training process or TPU device user already exists on worker ${WORKER_ID}" >&2
  exit 1
fi

printf -v COMMAND ' %q' "$@"
UPLOAD="for target in \"$LOG\" \"$LOG.exit\"; do [[ -e \"\$target\" ]] || continue; \
destination=\"$URI\"; [[ \"\$target\" == *.exit ]] && destination=\"$URI.exit\"; \
gcloud storage cp \"\$target\" \"\$destination\" >/dev/null 2>&1 || gsutil -q cp \"\$target\" \"\$destination\" || true; done"
setsid nohup bash -c "${COMMAND# } ; echo \$? > \"$LOG.exit\"" >"$LOG" 2>&1 </dev/null &
PID=$!
setsid nohup bash -c "while kill -0 $PID 2>/dev/null; do sleep $SYNC_SECONDS; $UPLOAD; done; $UPLOAD" \
  >"$LOG.sync" 2>&1 </dev/null &
echo "worker=${WORKER_ID} pid=${PID} log=${LOG} gcs=${URI}"
