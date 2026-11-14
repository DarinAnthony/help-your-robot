# help-your-robot

When should a robot ask for help? A CS 238 project on the identified DROID station in the SRC kitchen twin.
The research plan is [`PLAN.md`](PLAN.md); this file covers setup and what has been checked.

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
| `scripts/contact_sheet.py` | Frames across time from episode videos as one PNG, for a quick look |
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
`stanford/classes/cs238/data/shelf_pnp_droid_20261113`):

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

TidyBot drawer opening (`$MLSPACES_ARTIFACT_ROOT/collect/datagen/open_planner_v1/<run>`, sessions in
`$MLSPACES_ARTIFACT_ROOT/sessions/<run>`; on Drive under the run's name):

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

Then push any run directory. It is copied under `gdrive:stanford/classes/cs238/data/<run name>`, verified by size and
MD5, and nothing on Drive is deleted. `HELP_DRIVE_ROOT` and `HELP_DRIVE_REMOTE` override the destination; the script
refuses a root rclone cannot see rather than creating a duplicate (`HELP_DRIVE_CREATE_ROOT=1` creates it):

```bash
source env.sh && scripts/push_to_drive.sh $MLSPACES_ARTIFACT_ROOT/smoke/datagen/open_planner_v1/debug/droid_open_debug1
```

## Open problems

- Which station pose is right needs a real frame overlaid on the render (the D435 and ZED 2i views).
- The stock model is the FR3 with the Panda's working joint range (main's `droid_panda`); the identified torque plant
  (`droid_panda.TorqueResponse`) is not used yet.
- π0.5 training needs a CUDA GPU or TPU; it does not run on this Mac.
