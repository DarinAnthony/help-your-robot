#!/bin/bash
# Copy a Colab run that writes to local disk (pi05_train_colab.ipynb with USE_DRIVE = False) to Drive with rclone.
#
#   scripts/pull_ckpt.sh https://<name>.trycloudflare.com help_shelf_pi05_lora run1 [once]
#
# The notebook serves its ROOT (assets/, logs/, checkpoints/) at that URL. Every 5 minutes this copies assets/ and
# logs/, and the newest checkpoint the notebook marked finished (checkpoints/<cfg>/<exp>/.<step>.done) that is not on
# Drive yet; once it is complete there, the older checkpoint on Drive is removed. It then writes manifest.json next to
# it ({"step", "files": {path under pi05/: Drive file id}}) and prints its id: the notebook's RESTORE, which fetches
# that checkpoint through Drive's public link on a fresh runtime. Data flows Colab -> this Mac -> Drive, nothing is
# kept here.
set -uo pipefail
URL="${1:?usage: scripts/pull_ckpt.sh <tunnel url> <config> <exp> [once]}"; CFG="${2:?config}"; EXP="${3:?exp}"
RCLONE="${RCLONE:-$HOME/micromamba/envs/tools/bin/rclone}"
DEST="${HELP_DRIVE_REMOTE:-gdrive}:${HELP_DRIVE_PI05:-stanford/classes/cs238/help-your-robot/pi05}"
SRC=":http,url='$URL':"
C="checkpoints/$CFG/$EXP"
# python's http.server has no range requests: one stream per file.
rc() {
  "$RCLONE" --multi-thread-streams 0 --retries 5 --low-level-retries 20 "$@" 2>&1 | grep -v 'NOTICE\|client_id\|common hash'
  return "${PIPESTATUS[0]}"
}

pull_once() {
  rc copy "${SRC}assets" "$DEST/assets" -q
  rc copy "${SRC}logs" "$DEST/logs" -q
  local last id old
  last=$("$RCLONE" lsf "${SRC}$C" 2>/dev/null | sed -n 's/^\.\([0-9]*\)\.done$/\1/p' | sort -n | tail -1)
  [ -n "$last" ] || return 0
  "$RCLONE" lsf "$DEST/$C/.$last.done" >/dev/null 2>&1 && return 0
  echo "$(date '+%F %T') copying step $last to Drive"
  rc copy "${SRC}$C/$last" "$DEST/$C/$last" -q
  rc check "${SRC}$C/$last" "$DEST/$C/$last" --size-only --one-way -q || { echo "step $last incomplete, will retry"; return 0; }
  "$RCLONE" touch "$DEST/$C/.$last.done" 2>/dev/null
  for old in $("$RCLONE" lsf "$DEST/$C" --dirs-only 2>/dev/null | tr -d / | grep -E '^[0-9]+$'); do
    [ "$old" = "$last" ] || { rc purge "$DEST/$C/$old"; rc deletefile "$DEST/$C/.$old.done"; }
  done
  "$RCLONE" lsjson -R --files-only "$DEST" --include "assets/**" --include "$C/$last/**" 2>/dev/null \
    | python3 -c "import json, sys; print(json.dumps({'step': $last, 'files': {f['Path']: f['ID'] for f in json.load(sys.stdin)}}))" \
    | "$RCLONE" rcat "$DEST/$C/manifest.json" 2>/dev/null
  id=$("$RCLONE" lsjson "$DEST/$C/manifest.json" 2>/dev/null | python3 -c "import json, sys; print(json.load(sys.stdin)[0]['ID'])")
  echo "$(date '+%F %T') on Drive: step $last; RESTORE = '$id'"
}

if [ "${4:-}" = once ]; then pull_once; exit; fi
while true; do pull_once; sleep 300; done
