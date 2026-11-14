#!/usr/bin/env bash
# The identified TidyBot opening drawer/glass_counter__stack_b_3_front with base_pull (README, "Robots and tasks"),
# in the installed twin or in one of scripts/make_drawer_variants.py's copies.
#   scripts/collect_drawer_open.sh <run_name> <episodes> <seed> [variant]
# variant: a name built by make_drawer_variants.py (hard, locked, ...); none uses the nominal twin. Runs through
# run_pipeline_hooked.py, which pins the variant's drawer values over xemb's arena Drawer randomization.
set -euo pipefail
source "$HOME/research/help!/env.sh"
RUN=$1; N=$2; SEED=$3; VARIANT=${4:-}
if [[ -n $VARIANT ]]; then
  export SRC_KITCHEN_TWIN_DIR="$MLSPACES_ARTIFACT_ROOT/src_kitchen_twin_drawer_$VARIANT"
  [[ -f $SRC_KITCHEN_TWIN_DIR/variant.json ]] || { echo "no variant $VARIANT (scripts/make_drawer_variants.py)"; exit 1; }
fi
KG='"grasp_seat": "seated_v2", "grasp_close_mode": "until_settled", "gripper_close_duration": 1.5, "grasp_hold_min_gap": 0.0025, "mpc_time_cap": "replans_v1", "arm_velocity_limit_reference": "previous_command", "mpc_target_reference": "previous_target", "base_pull": {}'
# $HELP_FAULT (helpsim/xemb_hooks.py): faults for bad trajectories, e.g. '{"hidden_drawer": true}' with a variant,
# '{"grip_scale": 0.2}' or '{"stop_fraction": 0.3}'. The run's settings go to <run>/help_run.json.
RUN_DIR="$MLSPACES_ARTIFACT_ROOT/collect/datagen/open_planner_v1/$RUN"
mkdir -p "$RUN_DIR"
printf '{"run": "%s", "episodes": %s, "seed": %s, "variant": "%s", "fault": %s}\n' "$RUN" "$N" "$SEED" "$VARIANT" "${HELP_FAULT:-"{}"}" > "$RUN_DIR/help_run.json"
cd "$XEMB"
exec $PY ${PROBE:-} "$HOME/research/help!/scripts/run_pipeline_hooked.py" --task_type open --scene_dataset src_kitchen_twin --robot mobile_unified_arm \
  --variant_dir "$MLSPACES_ARTIFACT_ROOT/robots/tidybot_xarm7_measured" --planner kino --plan_mode mpc \
  --target_joint_kind slide --samples_per_house "$N" --num_workers 1 --seed "$SEED" --measured_command_path \
  --completion_mode return_to_initial --no_filter \
  --task_sampler_config_json '{"pickup_obj_name":"drawer/glass_counter__stack_b_3_front","exact_robot_base_pose":[7.76149,-1.73157,0.0,0.99995881,0.0,0.0,-0.00907571],"check_exact_robot_base_pose_visibility":false}' \
  --policy_config_json "{$KG}" --output_root "$MLSPACES_ARTIFACT_ROOT/collect" --run_name "$RUN"
