#!/bin/bash
# Copy a local folder of collected trajectories to Google Drive with rclone, then verify it.
#
#   source env.sh
#   scripts/push_to_drive.sh <local dir> [<path under $HELP_DRIVE_ROOT>]
#   scripts/push_to_drive.sh $MLSPACES_ARTIFACT_ROOT/smoke/datagen/open_planner_v1/debug/droid_open_debug1
#       -> gdrive:stanford/classes/cs238/data/droid_open_debug1
#
# `rclone copy` only adds and updates files on Drive; it never deletes there, so re-running after more collection
# uploads only what is new. Setup (once): see README "Google Drive".
set -euo pipefail

SRC="${1:?usage: scripts/push_to_drive.sh <local dir> [<drive subpath>]}"
[[ -d "$SRC" ]] || { echo "not a directory: $SRC" >&2; exit 2; }
RCLONE="${RCLONE:-$HOME/micromamba/envs/tools/bin/rclone}"
REMOTE="${HELP_DRIVE_REMOTE:-gdrive}"
ROOT="${HELP_DRIVE_ROOT:-stanford/classes/cs238/data}"
DEST="$REMOTE:$ROOT/${2:-$(basename "$SRC")}"

"$RCLONE" listremotes | grep -qx "$REMOTE:" || {
  echo "rclone remote '$REMOTE' is not configured; see README 'Google Drive'" >&2; exit 2; }
# The root must already exist: under scope=drive.file rclone cannot see folders it did not create, and would make a
# second folder of the same name instead. HELP_DRIVE_CREATE_ROOT=1 creates it deliberately.
if [[ "${HELP_DRIVE_CREATE_ROOT:-0}" != 1 ]] && ! "$RCLONE" lsf "$REMOTE:$ROOT" --max-depth 1 >/dev/null 2>&1; then
  echo "Drive folder $REMOTE:$ROOT not visible to rclone. If it exists, reconnect with scope=drive (README" \
       "'Google Drive'); to create it, set HELP_DRIVE_CREATE_ROOT=1" >&2
  exit 2
fi

echo "[push] $SRC -> $DEST"
"$RCLONE" copy "$SRC" "$DEST" --progress --transfers 8 --checkers 16 --drive-chunk-size 64M \
  --exclude '.DS_Store' --exclude '__pycache__/**'
# One-way check: every local file is on Drive with the same size and MD5.
"$RCLONE" check "$SRC" "$DEST" --one-way --exclude '.DS_Store' --exclude '__pycache__/**'
echo "[push] done: $DEST"
