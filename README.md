# help-your-robot

When should a robot ask for help? A CS 238 project on the identified DROID station in the SRC kitchen twin.
The research plan is [`PLAN.md`](PLAN.md) and the experiment (hypotheses, evaluation protocol, baselines, data it needs) is
[`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md); this file covers setup, what has been checked and the collected data.

## Layout

| Path | What |
|---|---|
| `env.sh` | Paths and environment for every command (`source env.sh`, then use `$PY`) |
| `helpsim/envs/kitchen_station.py` | The installed twin with the DROID at a measured station pose and the record's cameras |
| `scripts/station_check.py` | M0 check: penetration at home, renders from the station's cameras |
| `scripts/collect_drawer_open.sh` | The TidyBot drawer-opening collection, on the nominal drawer or a variant |
| `scripts/make_drawer_variants.py` | Hard and locked copies of the twin's demo drawer, with a push test of each |
| `scripts/run_pipeline_hooked.py`, `helpsim/xemb_hooks.py` | xemb's `run_pipeline.py` with help!'s hooks: the variant's drawer values, the TidyBot's own cameras for the preview videos |
| `scripts/export_drawer_sessions.sh` | Drawer episodes as hub sessions through the TidyBot's calibrated cameras (xemb's `session_export`) |
| `scripts/collect_shelf_pnp.py`, `helpsim/tasks/shelf_pnp.py` | The DROID shelf pick and place: rendered episodes, fault modes, and counterfactual action trees (`--tree`) |
| `scripts/drawer_attempts.py` | Per-pull attempt records of drawer runs from the H5 measured streams (`<run>/attempts.jsonl`) |
| `scripts/tree_summary.py` | Outcome-defined difficulty of every shelf tree start (`<tree root>/difficulty.jsonl`) |
| `scripts/contact_sheet.py` | Frames across time from episode videos as one PNG, for a quick look |
| `scripts/to_lerobot.py`, `helpsim/openpi_help.py`, `scripts/openpi_run.py`, `notebooks/pi05_train_colab.ipynb` | pi0.5: LeRobot datasets (train and 5% `_val`), the openpi LoRA configs, norm stats / training / open-loop check / policy server, and the Colab notebook (TASK_SPEC.md §7) |
| `helpsim/pi05_shelf.py`, `scripts/pi05_rollouts.py` | pi0.5 in the shelf sim against the policy server: attempt monitor (TASK_SPEC.md §4), reset and retry, help decisions, continue/reset trees; `replay-test` checks the monitor on expert episodes |
| `helpsim/pi05_drawer.py`, `scripts/run_drawer_pi05.sh` | pi0.5 in the drawer sim: xemb's collection pipeline with a pi0.5 TidyBot policy in place of the planner |
| `helpsim/help/`, `scripts/help_eval.py` | The help study: exact scoring on trees, value functions, baselines, belief POMDP; reports in `reports/help_eval/<family>/` |
| `scripts/embed_trees.py` | DINOv2 ViT-S/14 embeddings of every tree node's frames, for the V1 image value function |
| `pi05/` | π0.5 training code copied from xemb-vla (see [`pi05/PROVENANCE.md`](pi05/PROVENANCE.md)) |
| `data/` | Generated outputs (git-ignored) |

## Dependencies

- **xemb-vla** (private, `BoAi01/xemb-vla`) as a library, never copied except `pi05/`. `env.sh` points `$XEMB` at the
  worktree `~/research/xemb-vla-kitchen`, local branch `help/kitchen-latest`: `origin/main` (`c71a1199b`) plus four
  commits from `feat/src-kitchen-twin-axis-aligned` that main lacks (one level island countertop; linear textures and
  light levels fit to the DROID cameras; the lighting study note; the D435 exposure anchor). Not taken from that
  branch: its object commits and its 2026-11-09 DROID record (main has newer versions of both) and "Exposure looks"
  (opt-in render looks; conflicts with main's refactored datagen files).
- **HomeBody** `c09fcf39` at `~/research/third_party/homebody`: only the raw scan meshes. The twin build applies the
  real2sim corrections committed in xemb-vla (`scripts/analysis/homebody_kitchen/reference/`: scan corrections,
  front edits, taped interiors, measured joint physics, colours, lighting, surround).

## Setup on macOS (Apple silicon), as done 2026-11-13

```bash
# micromamba as a standalone binary (Homebrew here is owned by another account)
mkdir -p ~/.local/bin && cd ~/.local && curl -Ls https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xvj bin/micromamba
MAMBA_ROOT_PREFIX=~/micromamba ~/.local/bin/micromamba create -y -n mlspaces -c conda-forge python=3.11 git-lfs ffmpeg

# xemb-vla worktree with the latest kitchen work
git -C ~/research/xemb-vla fetch origin
git -C ~/research/xemb-vla worktree add -b help/kitchen-latest ~/research/xemb-vla-kitchen origin/main
#   then cherry-pick ceadb8df8 dfe661490 d1ec960b4 4c8cab1f2 (conflicts resolved by hand in envs/src_kitchen/README.md,
#   tests/test_src_kitchen_twin_env.py, notes/README.md)

source env.sh
$PY -m pip install -e "$XEMB[mujoco]"
$PY -c "import nltk; [nltk.download(p, download_dir='$NLTK_DATA') for p in ['wordnet2022', 'wordnet', 'omw-1.4', 'punkt']]"

git clone https://github.com/Stanford-TML/homebody.git "$HOMEBODY_ROOT" && git -C "$HOMEBODY_ROOT" checkout c09fcf39
cd "$XEMB" && $PY -m scripts.analysis.homebody_kitchen.twin --homebody-root "$HOMEBODY_ROOT" --install   # ~6-9 min
```

`wordnet2022` is required: xemb refuses to download corpora at runtime.

## Checked

| Check | Command | Result |
|---|---|---|
| Twin install | `install.json` | frame `levelled_axis_aligned`, textures `linear`, no environment fill, 61 parts, 34 open/close targets |
| Twin tests | `cd $XEMB && $PY -m pytest tests/test_src_kitchen_twin_env.py` | 80 passed, 1 skipped, 2 failed; the same 2 fail on plain `main` (`test_visual_split_keeps_the_collision_model`, `test_joint_band_scales_part_joints_around_their_installed_values`) |
| Stock DROID open smoke | `cd $XEMB && $PY scripts/datagen/run_pipeline.py --task_type open --scene_dataset src_kitchen_twin --robot droid --planner kino --plan_mode mpc --samples_per_house 2 --num_workers 1 --seed 0 --output_root $MLSPACES_ARTIFACT_ROOT/smoke --run_name droid_open_stock` | **0 of 12 opens as is.** xemb's arm rate limiter, with its default `arm_velocity_limit_reference: measured_qpos`, clips each command to 0.066 rad from the measured joints; the arm then moves at about 0.47 rad/s while the approach is timed for 1 rad/s, so every grasp closed 8-36 cm short. Gravity compensation does not change it. With `--policy_config_json '{"arm_velocity_limit_reference": "previous_command", "mpc_time_cap": "replans_v1"}'`: **12 of 13 episodes opened** (both fridge doors, 7 cabinet doors, 3 drawers; run `droid_open_prevcmd_replans_n12`, about 2 h with other simulations running); every grasp holds (TCP error 0.4 mm) and the one failure lost the handle mid-pull (`sorting_cabinet__front_00`). `replans_v1` replaces the wall-clock MPC budget, which ran out on 5 of 7 openings with other simulations loading the Mac |
| Station | `$PY scripts/station_check.py --out data/station_check` | both station poses collision-free at home; the ZED 2i view shows faucet, sinks, range, drawer stacks and dishwasher, the landmarks the record's overlay was checked on |

## Stations

| Key | Arm root (x, y, z, yaw) | Source |
|---|---|---|
| `droid_20261109` (default) | 6.5530, -2.2444, 0.867, 85.95° | `droid_panda_20261109`: ICP of ZED 2i + D435 depth against the kitchen-branch twin |
| `droid_real_v1` | 6.7159, -2.4872, 0.862, 93.57° | main's `SWEEP_BASES["droid_real_v1"]` (2026-10-29 episodes, edge chamfer) |

The two differ by about 30 cm and 7.6°. Main's sweep and population code still uses `droid_real_v1`.

## Robots and tasks

`$PY scripts/build_robots.py` builds both identified robots from xemb's `task_set.DEPLOYMENT_ROBOTS` into
`$MLSPACES_ARTIFACT_ROOT/robots/<name>/` (`model.xml`, `params.json`).

| Robot | Task | Where | Recipe |
|---|---|---|---|
| `droid_panda` (stock DROID on the real Panda's working joint range, arm gravity-compensated) | Shelf pick and place | The door-session cart `droid_corner_20261029` (8.188, 0.379, 0.848, yaw 40.6°); `upper_return`'s middle shelf (z 1.5065), the reachable front half (x 8.975-9.03, y 0.40-0.74) | help!'s scripted expert, `helpsim/tasks/shelf_pnp.py` |
| `tidybot_xarm7_measured` (rig `tidybot_xarm7_20261111`) | Drawer opening, base-driven | `drawer/glass_counter__stack_b_3_front` from (7.76149, -1.73157, yaw -1.04°) | xemb's kino planner on the identified command path (`--measured_command_path`, your 2026-10-27 handoff recipe with `base_pull`): `scripts/collect_drawer_open.sh` |

Why these choices:
- **The DROID cannot reach any upper cabinet from the island station** (3.7-4.4 m away; `scripts/shelf_reach.py`). The
  door-session cart reaches 5 of 15 cells on the middle shelf, all at the front; none on the top shelf.
- **xemb's pick-and-place samplers** can neither pin the base nor use cabinet shelves (they are not placement
  surfaces), so the shelf task is help!'s own scripted expert: a level approach with the jaws horizontal, Cartesian
  waypoints by IK, the stock position servos, and up to 3 attempts. A retry swings the approach 15-30°, which turns
  the jaws across a long object, and perturbs the grasp point.
- **Gravity compensation is required** (`kitchen_station.build_spec(gravcomp=True)`). Without it the arm sags about 1 cm,
  the open fingers press 450 N into the shelf, and the servos stall at the Panda's 87 N·m limit, 10 cm short of the
  target. (The stock DROID open smoke's 0/12 looked the same but has another cause: xemb's arm rate limiter with its
  default `arm_velocity_limit_reference: measured_qpos` leaves the arm halfway along each segment;
  `--policy_config_json '{"arm_velocity_limit_reference": "previous_command", "mpc_time_cap": "replans_v1"}'` fixes
  it; see the smoke row above.)
- **The TidyBot on stock servos failed** (0 of 12; the demo stance is 0.92 m from the handle and IK fails for the
  pregrasp). On the identified command path with the base pulling it opened 2 of 2 (drawer +28 cm, base travel 34 cm).
  xemb has made that path reproduction-only for its *population* collections; plain `run_pipeline.py` runs it.

Shelf pick-and-place outcomes per attempt: `success`, `unreachable` (with the failing phase), `grasp_miss`, `slip`,
`place_miss`, `off_shelf`. Pilot (6 episodes each): tomato 6/6; egg, drumstick, wooden ball, orange 5/6; banana,
waffle, sponge, duster and tape mostly fail (too wide for the 85 mm jaws when lying flat, as on the real gripper);
metal spoon 0/6.

Object split (`scripts/collect_shelf_pnp.py`):

| Split | Objects | Why |
|---|---|---|
| `train` | tomato, egg, drumstick, wooden ball, banana, waffle, sponge, duster, tape | Successes and failures, so a success predictor has both |
| `heldout_in_distribution` | orange | Round and sized like the tomato, ball and egg: should mostly be attempted |
| `heldout_hard` | metal spoon | Thin and flat on the shelf: should trigger asking |

Not used: the wooden spoon (the same failure category as the held-out spoon) and the plate (wider than the shelf).

### Drawer variants

`$PY scripts/make_drawer_variants.py hard locked` builds `$MLSPACES_ARTIFACT_ROOT/src_kitchen_twin_drawer_<name>/`: the
installed twin linked entry by entry, an edited `scene.xml`, and `variant.json` (the settings, the nominal values and a
push test: a force on the slide held for 1 s).

| Variant | Change to `drawer/glass_counter__stack_b_3_front` | Push test | Opens |
|---|---|---|---|
| nominal | none (slide frictionloss U(1, 2) N, see below) | 20 N moves it 2 cm, 30 N opens it | 20/20 |
| `hard` | slide frictionloss 6.5 N | 20 N moves it 1 cm, 30 N opens it | 15/20 |
| `locked` | a tendon limit stops the slide at 8 mm (a latch) | holds 200 N at 9.8 mm | 0/10 |

**xemb overwrites the scene file's drawer physics.** Its arena `Drawer` (`molmo_spaces/env/arena/drawer.py`) draws,
every episode, the slide's frictionloss U(1, 2) N, damping U(0.01, 0.1), armature U(0, 0.1) and every geom's friction
U(0.1, 0.3). Neither the twin's identified values (frictionloss 0.4 N, damping 4) nor a variant's reach the simulation;
the nominal collection ran on xemb's draw. `scripts/run_pipeline_hooked.py` runs `run_pipeline.py` with
`helpsim/xemb_hooks.py`, which applies `variant.json`'s values after that draw (xemb is not edited). Tendons are not
touched, so the latch works either way. Before the hook, frictionloss up to 400 N in the scene file still opened.

The hard drawer's 6.5 N comes from a sweep with the hook (seed 3): 5 N opened 7/8, 6.5 N 3/6, 8 N 0/8. The TidyBot's
base servos (kp 641340, no force limit) pull far harder than 6.5 N; what fails is the kino planner, which rehearses the
pull in simulation and abandons the episode ("planner found no trajectory") when its rehearsal does not open the
drawer. The episode is still saved, about 150 steps of approach and return. So almost every failure is a planner
refusal, decided with the simulator's privileged state, and few are executed failures (the handle slipping from the
jaws while the base keeps backing away): hard 4 refusals and 1 executed failure, locked 10 refusals. Outcomes on the
hard drawer depend on the planner's random stream: the same seeds opened 6/20 in a first round, before the camera
hook shifted that stream, and 15/20 in the collection below.

### Bad trajectories: common failure modes

For the value function, both tasks are also collected with faults that cause the failures a real robot commonly has.
Each fault is set for a whole run, so every episode's cause is known; the fault is not in the recorded state.

Shelf (`scripts/collect_shelf_pnp.py --mode`, `helpsim/tasks/shelf_pnp.py` `FAULTS`). Calibration on the four easy
objects, 8 episodes each (nominal: 80/100 on the 20261113 seeds):

| Mode | Fault | Success | Failures it adds |
|---|---|---|---|
| `pose_error` | object-position estimate off by N(0, 1.5 cm) in x, y (0.75 cm in z), redrawn each attempt | 19/32 | missed and edge grasps, slips, knocked objects |
| `place_error` | target estimate off by N(0, 4 cm) in x, y, redrawn each attempt | 13/32 | misplacements |
| `weak_grip` | gripper force limit × 0.05 | 14/32 | missed grasps, slips under load |

A slippery object (pad friction 0.08 with contact priority) still held the light kitchen objects (21/32, few slips),
so it is not a mode. A first version put the low friction on the object, which also made it slide along the shelf
and turned its failures into unreachable grasps.

Drawer (`$HELP_FAULT`, `helpsim/xemb_hooks.py`):

| Fault | `$HELP_FAULT` | Smoke | What happens |
|---|---|---|---|
| hidden stuck drawer | `{"hidden_drawer": true}` with variant `stuck` (frictionloss 20 N) | 0/3 | the planner rehearses on a nominal drawer, commits, and the handle slides out of the jaws as the base backs away |
| hidden locked drawer | `{"hidden_drawer": true}` with variant `locked` | 0/3 | the same against the latch |
| weak grip | `{"grip_scale": 0.02}` | 1/3 | the handle slips during the pull (0.05 and 0.2 still open every time) |
| pull stops early | `{"stop_fraction": 0.3}` | 0/2 | the base stops after 30% of the planned pull |

Without `hidden_drawer`, the planner sees the variant and refuses to pull (the drawer variants above). With it, every
smoke episode planned an opening and failed while executing it. xemb keeps the model across episodes, so the hook
hides the drawer again (and restores the grip) when the policy resets for the next episode. `make_drawer_variants.py
--custom stuck --frictionloss 20` builds the stuck drawer.

### Sequential data: drawer re-pulls and shelf action trees

The help policy decides at every attempt, so the data needs second and third attempts from the states real failures
leave, and the evaluation needs to know what every choice would have led to (EXPERIMENT_PLAN.md §4).

**Drawer re-pulls** (`HELP_REPULLS=2`, `helpsim/xemb_hooks.py`). xemb retries a pull only when its own rehearsal predicts
a slip, so before this every drawer episode was one pull. With re-pulls, a pull that ends with the drawer not open is
followed by a release and return: the jaws open with the arm held (1.2 s), the hand backs off 8 cm, the arm goes home,
the base drives back to where the pull started, and the policy's retry re-plans grasp and pull, up to 2 more times.
`collect_drawer_open.sh` then raises the episode's step limit to 750 (a failed pull plus its return is ~220 steps). A
hidden drawer is hidden again for every re-plan. `"stop_pulls": 1` with `stop_fraction` makes the early stop a one-off
that a re-pull recovers from. Two things the smoke runs caught on the way:

- After a failed pull the base had backed ~0.3 m away, so the re-grasp found no feasible grasp; hence the base's return.
- xemb's own slip release opens the jaws for 0.25 s, shorter than the identified jaws take, and its retreat then
  re-commands them closed: the arm going home dragged a stopped drawer open to 25 cm and the task scored it a success.
  The release now keeps the gripper commanded open on every step.

Smoke (`smk6_*`, deleted): a locked drawer, three pulls each stopping at 8.5 mm with the base returning in between; a
one-off early stop, pull 1 at 6.3 cm, re-pull opened to 30 cm. `scripts/drawer_attempts.py <run>` splits each episode
into pulls by `obs/extra/policy_num_retries` and records each pull's drawer travel (from the 250 Hz measured streams),
base travel and outcome.

**Shelf action trees** (`collect_shelf_pnp.py --tree`, `ShelfPnP.run_tree`). From each start state, every
attempt / adjust sequence up to 3 attempts: at each decision point the full physics state is saved (`mj_getState`,
`mjSTATE_INTEGRATION`, so a restore replays exactly) and both continuations are run, each with its own random stream;
decision-point frames are rendered at every node. Up to 14 attempts per start, ~10 s each. The trees use the rendered
episodes' seeds, so each tree shares its start with a video episode. Without injected noise (nominal, weak grip) the
tree's first attempt reproduces the episode's in 702/715 starts; with pose or place noise the branches draw their own
noise, so the first attempt matches the episode only in 372/550. Checked: the same tree twice gives identical outcomes
and object poses.

### TidyBot cameras

**The TidyBot's camera output is the hub session**, written by xemb's `molmo_spaces.robots.measured.session_export`
(`scripts/export_drawer_sessions.sh`). It replays each episode's measured streams in the twin and renders the mast
RealSense (`realsense`) and the wrist Arducam (`arducam`) through their calibrated lenses at 1280x720, with the camera
pose, lens, exposure, white balance and stream timing drawn per episode around the calibration. Hub sessions are what
remact's LeRobot converter reads. Successes and failures are exported alike; the manifest's outcome is the episode's.

The videos `run_pipeline.py` writes during collection are previews. Stock, it gives the mobile arm the UR5e's camera
system: `exo_camera_1` at a fixed offset from the arm root, which on the TidyBot looks across the kitchen and sees
neither robot nor drawer, and the wrist camera at the UR5e's 56.74°. The hook renders `exo_camera_1` from the robot's
`mast_camera` and the wrist camera at its MJCF field of view instead. `tidybot_open_20261113` (below) was collected
before the hook, so its videos are the UR5e stand-ins; its trajectories still export to sessions.

### Collected data

Shelf pick and place, `data/shelf_pnp/20261113` (25 episodes per object; on Drive as
`stanford/classes/cs238/help-your-robot/data/shelf_pnp_droid_20261113`):

| Split | Object | Success | Succeeded on a retry |
|---|---|---|---|
| train | wooden ball | 22/25 | 2 |
| train | tomato | 21/25 | 3 |
| train | egg | 19/25 | 4 |
| train | drumstick | 18/25 | 2 |
| train | waffle | 7/25 | 0 |
| train | sponge | 6/25 | 0 |
| train | duster | 4/25 | 1 |
| train | banana | 1/25 | 1 |
| train | tape | 0/25 | 0 |
| heldout_in_distribution | orange | 21/25 | 2 |
| heldout_hard | metal spoon | 0/25 | 0 |

Shelf pick and place in four expert modes, `data/shelf_pnp/20261114/<mode>` (seed0 20000 in every mode, so episode
*i* of an object starts from the same state and target in each; rendered; 1,265 episodes):

| Object | nominal | pose_error | place_error | weak_grip |
|---|---|---|---|---|
| egg | 38/40 | 15/25 | 11/25 | 22/25 |
| tomato | 35/40 | 12/25 | 12/25 | 1/25 |
| wooden ball | 29/40 | 18/25 | 9/25 | 17/25 |
| drumstick | 22/40 | 11/25 | 4/25 | 6/25 |
| waffle | 13/40 | 12/25 | 8/25 | 10/25 |
| duster | 12/40 | 9/25 | 6/25 | 4/25 |
| sponge | 6/40 | 5/25 | 2/25 | 0/25 |
| banana | 0/40 | 0/25 | 0/25 | 8/25 |
| tape | 0/40 | 0/25 | 0/25 | 0/25 |
| orange (held out, in distribution) | 32/40 | 15/25 | 14/25 | 20/25 |
| metal spoon (held out, hard) | 0/40 | 1/25 | 0/25 | 0/25 |
| final outcomes | 187 success, 142 unreachable, 80 grasp miss, 19 slip, 7 off shelf, 5 place miss | 98 / 92 / 63 / 16 / 3 / 3 | 66 / 93 / 47 / 14 / 17 / 38 | 88 / 83 / 80 / 12 / 5 / 7 |

`$PY scripts/collect_shelf_pnp.py --out data/shelf_pnp/20261114/<mode> --mode <mode> --episodes <40 nominal, 25 otherwise> --seed0 20000 --workers 6`.

TidyBot drawer, single pull with faults, 20261114 (seeds 500-1100; failure-mode runs of the section above):

| Run | Drawer / fault | Opened | First-pull drawer travel |
|---|---|---|---|
| `tidybot_open_nominal_20261114` | nominal | 30/30 | 0.29-0.30 m |
| `tidybot_open_hidden_hard_20261114` | `hard`, hidden | 7/20 | 0.034 m when it fails |
| `tidybot_open_hidden_stuck_20261114` | `stuck`, hidden | 0/20 | 0.019-0.020 m |
| `tidybot_open_hidden_locked_20261114` | `locked`, hidden | 0/20 | 0.008 m |

The weak-grip and stop-early runs of that day were stopped part way and are superseded by the re-pull runs.

Shelf action trees, `data/shelf_pnp/20261114_tree/<mode>` (the 1,265 starts above, 12,892 attempts; on Drive as
`shelf_pnp_droid_20261114_tree`). Each start's hidden difficulty from its tree (`scripts/tree_summary.py`): easy, the
first attempt succeeds; recoverable, a later attempt within 3 does; infeasible, none does:

| Mode | easy | recoverable | infeasible |
|---|---|---|---|
| nominal | 165 | 84 | 191 |
| pose_error | 79 | 65 | 131 |
| place_error | 53 | 63 | 159 |
| weak_grip | 77 | 40 | 158 |

Recoverable starts split between the actions: the shortest successful path starts with an adjust in 178 of 252 (`D`
alone 56, `DA` 78), so attempt and adjust are not interchangeable. Per object, the nominal mix runs from all easy or
recoverable (wooden ball 24/16/0, egg 33/6/1) to all infeasible (tape 0/0/40, metal spoon 0/2/38).

TidyBot drawer with re-pulls, 20261115 (`HELP_REPULLS=2`, step limit 750, seeds 2000-2615; trajectories on Drive as
`drawer_open_tidybot_20261115/trajectories/<run>`). Pull sequences from `scripts/drawer_attempts.py`; `stuck` here
means the drawer travelled under 5 cm:

| Run | Drawer / fault | Opened | Pull sequences |
|---|---|---|---|
| `tidybot_open_rp_nominal_20261115` | nominal | 20/20 | success 20 |
| `tidybot_open_rp_transient_stop_20261115` | stop at 30 %, first pull only | 20/20 | short-success 20 (first pull 6.1-6.3 cm) |
| `tidybot_open_rp_hidden_medium_{a,b}_20261115` | `medium` (held out), hidden | 30/30 | success 22, stuck-success 6, stuck-stuck-success 2 |
| `tidybot_open_rp_hidden_hard_{a,b}_20261115` | `hard`, hidden | 32/40 | success 24, stuck-success 4, stuck-stuck-success 4, 3x stuck 8 |
| `tidybot_open_rp_hidden_stuck_20261115` | `stuck`, hidden | 0/25 | 3x stuck 25 (each under 2.1 cm) |
| `tidybot_open_rp_hidden_locked_20261115` | `locked`, hidden | 0/25 | 3x stuck 25 (8.5 mm each) |
| `tidybot_open_rp_weak_grip_20261115` | grip force x0.02 | 4/20 | success 4, stuck then abort 16 |

- Weak grip: after the first pull slips, the re-plan's grasp check simulates holding the handle with the weakened
  grip, finds no grasp that holds, and xemb ends the episode ("no reachable +holding grasp — fail fast"). The
  planner detects the fault itself, so these episodes have one pull and no second.
- Stuck: the gripper keeps the handle and the drawer resists, so the base sometimes stalls instead of backing off
  (base travel 0 for every second pull, 0-0.3 m for the others). Locked drawers let go of the hand, and the base
  backs off the full 0.29 m every time.
- `medium` (frictionloss 4.5) turned out closer to nominal than to hard: 22 of 30 first pulls open it. It remains the
  held-out drawer for EXPERIMENT_PLAN.md.

Sessions for the re-pull runs and the three hidden 20261114 runs are exported with `scripts/export_drawer_sessions.sh`
(about an hour per episode per process at 1280x720) to `$MLSPACES_ARTIFACT_ROOT/sessions/<run>`. The trajectories'
preview mp4s (mast and wrist) are there meanwhile.

TidyBot drawer opening (`$MLSPACES_ARTIFACT_ROOT/collect/datagen/open_planner_v1/<run>`, sessions in
`$MLSPACES_ARTIFACT_ROOT/sessions/<run>`; the four 20261113b runs on Drive as
`stanford/classes/cs238/help-your-robot/data/drawer_open_tidybot_20261113b/{trajectories,sessions}/<run>`):

| Run | Drawer | Command | Opened | Failures |
|---|---|---|---|---|
| `tidybot_open_nominal_20261113b` | nominal | `scripts/collect_drawer_open.sh tidybot_open_nominal_20261113b 20 3` | 20/20 | |
| `tidybot_open_hard_a_20261113b` | `hard` | `scripts/collect_drawer_open.sh tidybot_open_hard_a_20261113b 10 100 hard` | 10/10 | |
| `tidybot_open_hard_b_20261113b` | `hard` | `scripts/collect_drawer_open.sh tidybot_open_hard_b_20261113b 10 200 hard` | 5/10 | 4 refused, 1 executed |
| `tidybot_open_locked_20261113b` | `locked` | `scripts/collect_drawer_open.sh tidybot_open_locked_20261113b 10 300 locked` | 0/10 | 10 refused |
| `tidybot_open_20261113` | nominal | the recipe above, `--samples_per_house 20 --seed 3`, before the hooks | 20/20 | |

Each run's sessions: `scripts/export_drawer_sessions.sh <run> [variant]`. `tidybot_open_20261113` is on Drive as
`drawer_open_tidybot_20261113` with its UR5e stand-in videos.

| Run | Command | Layout |
|---|---|---|
| `data/shelf_pnp/20261113` | `$PY scripts/collect_shelf_pnp.py --out data/shelf_pnp/20261113 --episodes 25` | `<split>/<object>/ep_<seed>.npz` (per 15 Hz tick: arm joints and targets, gripper, TCP pose, object pose, phase, attempt; `frame_tick`), `ep_<seed>_{cam_cabinet,wrist_camera}.mp4` (5 Hz), `episodes.jsonl` (outcome of every attempt) |
| `$MLSPACES_ARTIFACT_ROOT/collect/datagen/open_planner_v1/tidybot_open_*` | `scripts/collect_drawer_open.sh` | xemb's H5: `house_0/trajectories_batch_*.h5`, one `traj_i` per episode (obs, actions, per-step success, the measured streams), and preview mp4s |
| `$MLSPACES_ARTIFACT_ROOT/sessions/tidybot_open_*` | `scripts/export_drawer_sessions.sh` | one hub session per episode, `<time>-open-<run>.<batch>-traj_<i>/` (`manifest.json` with the outcome, `frames.jsonl`, `controls.jsonl`, `arm_report.jsonl`, `<frame>-{realsense,arducam}.jpg`), and `export_<batch>.{jsonl,log}` |

`cam_cabinet` is a synthetic third-person view of the shelf. The record's ZED 2i is calibrated for the island
station, and from the cart it sees the counter.

## Google Drive

Collected trajectories go to Drive with rclone (`scripts/push_to_drive.sh`). rclone is in its own micromamba env:

```bash
MAMBA_ROOT_PREFIX=~/micromamba ~/.local/bin/micromamba create -y -n tools -c conda-forge rclone
```

Connect it once (opens a browser to sign in and approve). The data folder already exists in Drive, so rclone needs
full Drive access (`scope=drive`); `drive.file` would only see folders rclone itself created:

```bash
~/micromamba/envs/tools/bin/rclone config create gdrive drive scope=drive
# an existing drive.file remote: widen it and re-authorize
~/micromamba/envs/tools/bin/rclone config update gdrive scope drive && ~/micromamba/envs/tools/bin/rclone config reconnect gdrive:
```

Everything for this project lives in its own folder on the Stanford Google Drive,
`stanford/classes/cs238/help-your-robot/`: `data/` holds the collected runs and the LeRobot datasets (`data/lerobot/`),
and `pi05/` holds what the Colab notebook writes (norm stats, checkpoints, logs).

Then push any run directory. It is copied under `gdrive:stanford/classes/cs238/help-your-robot/data/<run name>`,
verified by size and MD5, and nothing on Drive is deleted. `HELP_DRIVE_ROOT` and `HELP_DRIVE_REMOTE` override the destination; the script
refuses a root rclone cannot see rather than creating a duplicate (`HELP_DRIVE_CREATE_ROOT=1` creates it):

```bash
source env.sh && scripts/push_to_drive.sh $MLSPACES_ARTIFACT_ROOT/smoke/datagen/open_planner_v1/debug/droid_open_debug1
```

## Open problems

- Which station pose is right needs a real frame overlaid on the render (the D435 and ZED 2i views).
- The stock model is the FR3 with the Panda's working joint range (main's `droid_panda`); the identified torque plant
  (`droid_panda.TorqueResponse`) is not used yet.
- π0.5 training needs a CUDA GPU or TPU; it does not run on this Mac.
