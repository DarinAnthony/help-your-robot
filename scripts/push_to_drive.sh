#!/bin/bash
# Copy a local folder of collected trajectories to Google Drive with rclone, then verify it.
#
#   source env.sh
#   scripts/push_to_drive.sh <local dir> [<path under $HELP_DRIVE_ROOT>]
#   scripts/push_to_drive.sh $MLSPACES_ARTIFACT_ROOT/smoke/datagen/open_planner_v1/debug/droid_open_debug1
#       -> gdrive:help-your-robot/trajectories/droid_open_debug1
#
# `rclone copy` only adds and updates files on Drive; it never deletes there, so re-running after more collection
# uploads only what is new. Setup (once): see README "Google Drive".
set -euo pipefail

SRC="${1:?usage: scripts/push_to_drive.sh <local dir> [<drive subpath>]}"
[[ -d "$SRC" ]] || { echo "not a directory: $SRC" >&2; exit 2; }
RCLONE="${RCLONE:-$HOME/micromamba/envs/tools/bin/rclone}"
REMOTE="${HELP_DRIVE_REMOTE:-gdrive}"
ROOT="${HELP_DRIVE_ROOT:-help-your-robot/trajectories}"
DEST="$REMOTE:$ROOT/${2:-$(basename "$SRC")}"

"$RCLONE" listremotes | grep -qx "$REMOTE:" || {
  echo "rclone remote '$REMOTE' is not configured; see README 'Google Drive'" >&2; exit 2; }

echo "[push] $SRC -> $DEST"
"$RCLONE" copy "$SRC" "$DEST" --progress --transfers 8 --checkers 16 --drive-chunk-size 64M \
  --exclude '.DS_Store' --exclude '__pycache__/**'
# One-way check: every local file is on Drive with the same size and MD5.
"$RCLONE" check "$SRC" "$DEST" --one-way --exclude '.DS_Store' --exclude '__pycache__/**'
echo "[push] done: $DEST"
