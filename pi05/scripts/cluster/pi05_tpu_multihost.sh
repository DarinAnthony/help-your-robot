#!/usr/bin/env bash
# Run one explicit command concurrently on every worker of an existing Cloud TPU VM.

set -euo pipefail

PROJECT="${PROJECT:-soe-iris-gcp}"
ZONE="${ZONE:-us-central2-b}"
TPU_NAME="${TPU_NAME:?set TPU_NAME to the Cloud TPU node name}"
REMOTE_USER="${REMOTE_USER:-$USER}"
REMOTE_REPO="${REMOTE_REPO:-/home/$REMOTE_USER/embodiment-scaling-manipulation-data}"
GCLOUD="${GCLOUD:-gcloud}"
GCLOUD_SSH_KEY_FILE="${GCLOUD_SSH_KEY_FILE:-}"
GCLOUD_KNOWN_HOSTS_FILE="${GCLOUD_KNOWN_HOSTS_FILE:-}"
GCLOUD_PLAIN_SSH="${GCLOUD_PLAIN_SSH:-0}"

if [[ "${1:-}" == "--" ]]; then
    shift
fi
if [[ $# -eq 0 ]]; then
    echo "usage: TPU_NAME=<vm> $0 -- <command> [args...]" >&2
    exit 2
fi

quote_command() {
    local quoted
    printf -v quoted ' %q' "$@"
    printf '%s' "$quoted"
}

remote_command="cd $(quote_command "$REMOTE_REPO") && exec$(quote_command "$@")"
ssh_args=()
if [[ "${GCLOUD_PLAIN_SSH}" == "1" ]]; then
    if [[ -z "${GCLOUD_SSH_KEY_FILE}" || -z "${GCLOUD_KNOWN_HOSTS_FILE}" ]]; then
        echo "GCLOUD_PLAIN_SSH=1 requires local key and known-host paths" >&2
        exit 2
    fi
    ssh_args+=(
        --plain
        --ssh-key-file="${GCLOUD_SSH_KEY_FILE}"
        --ssh-flag="-F/dev/null"
        --ssh-flag="-x"
        --ssh-flag="-i${GCLOUD_SSH_KEY_FILE}"
        --ssh-flag="-oIdentitiesOnly=yes"
        --ssh-flag="-oUserKnownHostsFile=${GCLOUD_KNOWN_HOSTS_FILE}"
        --ssh-flag="-oStrictHostKeyChecking=no"
    )
elif [[ -n "${GCLOUD_SSH_KEY_FILE}" ]]; then
    ssh_args+=(--ssh-key-file="${GCLOUD_SSH_KEY_FILE}")
elif [[ -n "${GCLOUD_KNOWN_HOSTS_FILE}" ]]; then
    ssh_args+=(
        --ssh-flag="-o"
        --ssh-flag="UserKnownHostsFile=${GCLOUD_KNOWN_HOSTS_FILE}"
    )
fi
"$GCLOUD" compute tpus tpu-vm ssh "$TPU_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --worker=all \
    "${ssh_args[@]}" \
    --command="$remote_command"
