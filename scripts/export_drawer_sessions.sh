#!/usr/bin/env bash
# The TidyBot drawer episodes as hub sessions, through the robot's calibrated cameras: xemb's
# molmo_spaces.robots.measured.session_export replays each episode's measured streams in the twin and renders the mast
# RealSense and the wrist Arducam through their lenses at 1280x720, with the camera pose, lens, exposure, white
# balance and stream timing drawn per episode around the calibration. This is the TidyBot's camera output; the
# videos run_pipeline writes during collection are previews (helpsim/xemb_hooks.py). Every episode is exported,
# successes and failures; the session's outcome is the episode's task success.
#   scripts/export_drawer_sessions.sh <run_name> [variant] [jobs]
# One process per trajectories file, `jobs` at a time (default 4). Sessions go to
# $MLSPACES_ARTIFACT_ROOT/sessions/<run_name>/, and each file's report (one JSON line per episode) to export_<batch>.jsonl there.
# session_export names a session <wall-clock second>-open-<run directory>-traj_<n>, and every file counts its
# trajectories from traj_0, so two files exported at once collide on the same second. Each file is exported through a
# link in its own run directory, <run_name>.<batch>, which goes into the session's name (the manifest's source.h5
# is that link, $OUT/.src/<run_name>.<batch>/house_0/<file>.h5; the file it points to is in the run's house_0).
set -euo pipefail
source "$HOME/research/help!/env.sh"
RUN=$1; VARIANT=${2:-}; JOBS=${3:-4}
TWIN=$SRC_KITCHEN_TWIN_DIR
[[ -n $VARIANT ]] && TWIN="$MLSPACES_ARTIFACT_ROOT/src_kitchen_twin_drawer_$VARIANT"
SRC="$MLSPACES_ARTIFACT_ROOT/collect/datagen/open_planner_v1/$RUN/house_0"
OUT="$MLSPACES_ARTIFACT_ROOT/sessions/$RUN"
mkdir -p "$OUT"
cd "$XEMB"
ls "$SRC"/*.h5 | xargs -P "$JOBS" -I{} sh -c '
  h5="$1"; b=$(basename "$h5" .h5)
  link="$3/.src/$4.${b#trajectories_}/house_0"; mkdir -p "$link"; ln -sf "$h5" "$link/"; h5="$link/$b.h5"
  "$PY" -m molmo_spaces.robots.measured.session_export --h5 "$h5" --robot-dir "$MLSPACES_ARTIFACT_ROOT/robots/tidybot_xarm7_measured" \
    --twin "$2" --out "$3" --seed 0 > "$3/export_$b.jsonl" 2> "$3/export_$b.log"
' _ {} "$TWIN" "$OUT" "$RUN" || echo "an export stopped early: see $OUT/export_*.log"
grep -h '"outcome"' "$OUT"/export_*.jsonl | $PY -c 'import sys, json, collections; c = collections.Counter(json.loads(l)["outcome"] for l in sys.stdin); print(dict(c))'
