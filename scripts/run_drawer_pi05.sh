#!/usr/bin/env bash
# pi0.5 opening drawer/glass_counter__stack_b_3_front with the TidyBot, set up exactly as scripts/collect_drawer_open.sh
# collects it (same scene, start pose, robot, cameras and success rule), the policy from $HELP_PI05_SOURCE
# (helpsim/pi05_drawer.py: a policy server URL, or replay:<h5>:<traj> to test without a model).
#   HELP_PI05_SOURCE=wss://<name>.trycloudflare.com scripts/run_drawer_pi05.sh <run_name> <episodes> <seed> [variant]
# A variant (scripts/make_drawer_variants.py) and $HELP_FAULT work as in collect_drawer_open.sh. Attempts, decisions
# and resets (helpsim/pi05_drawer.py): $HELP_PI05_ATTEMPTS (default 3), $HELP_PI05_DECIDE (reset | continue),
# $HELP_PI05_SAMPLES (pi0.5's spread at each decision point; default 0). $TASK_HORIZON: policy steps per episode
# (default 550 per attempt: a 25 s attempt is ~380 steps at the ~15 Hz policy step, a reset ~150). Then
# scripts/drawer_attempts.py <run> writes the run's attempts.jsonl.
set -euo pipefail
source "$HOME/research/help!/env.sh"
RUN=$1; N=$2; SEED=$3; VARIANT=${4:-}
: "${HELP_PI05_SOURCE:?set HELP_PI05_SOURCE (policy server URL or replay:<h5>:<traj>)}"
if [[ -n $VARIANT ]]; then
  export SRC_KITCHEN_TWIN_DIR="$MLSPACES_ARTIFACT_ROOT/src_kitchen_twin_drawer_$VARIANT"
  [[ -f $SRC_KITCHEN_TWIN_DIR/variant.json ]] || { echo "no variant $VARIANT (scripts/make_drawer_variants.py)"; exit 1; }
fi
KG='"grasp_seat": "seated_v2", "grasp_close_mode": "until_settled", "gripper_close_duration": 1.5, "grasp_hold_min_gap": 0.0025, "mpc_time_cap": "replans_v1", "arm_velocity_limit_reference": "previous_command", "mpc_target_reference": "previous_target", "base_pull": {}'
HORIZON=${TASK_HORIZON:-$(( ${HELP_PI05_ATTEMPTS:-3} * 550 ))}
RUN_DIR="$MLSPACES_ARTIFACT_ROOT/collect/datagen/open_planner_v1/$RUN"   # where the pipeline writes the run
mkdir -p "$RUN_DIR"
printf '{"run": "%s", "episodes": %s, "seed": %s, "variant": "%s", "fault": %s, "source": "%s", "exec": %s, "attempts": %s, "decide": "%s", "samples": %s, "task_horizon": %s}\n' \
  "$RUN" "$N" "$SEED" "$VARIANT" "${HELP_FAULT:-"{}"}" "$HELP_PI05_SOURCE" "${HELP_PI05_EXEC:-8}" "${HELP_PI05_ATTEMPTS:-3}" \
  "${HELP_PI05_DECIDE:-reset}" "${HELP_PI05_SAMPLES:-0}" "$HORIZON" > "$RUN_DIR/help_run.json"
cd "$XEMB"
exec $PY "$HOME/research/help!/scripts/run_pipeline_pi05.py" --task_type open --scene_dataset src_kitchen_twin --robot mobile_unified_arm \
  --variant_dir "$MLSPACES_ARTIFACT_ROOT/robots/tidybot_xarm7_measured" --planner kino --plan_mode mpc \
  --target_joint_kind slide --samples_per_house "$N" --num_workers 1 --seed "$SEED" --measured_command_path \
  --completion_mode stop --no_filter --task_horizon "$HORIZON" \
  --task_sampler_config_json '{"pickup_obj_name":"drawer/glass_counter__stack_b_3_front","exact_robot_base_pose":[7.76149,-1.73157,0.0,0.99995881,0.0,0.0,-0.00907571],"check_exact_robot_base_pose_visibility":false}' \
  --policy_config_json "{$KG}" --output_root "$MLSPACES_ARTIFACT_ROOT/collect" --run_name "$RUN"
