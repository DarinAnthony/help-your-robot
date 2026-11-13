# help-your-robot

When should a robot ask for help? A CS 238 project on the identified DROID station in the SRC kitchen twin.
The research plan is [`PLAN.md`](PLAN.md); this file covers setup and what has been checked.

## Layout

| Path | What |
|---|---|
| `env.sh` | Paths and environment for every command (`source env.sh`, then use `$PY`) |
| `helpsim/envs/kitchen_station.py` | The installed twin with the DROID at a measured station pose and the record's cameras |
| `scripts/station_check.py` | M0 check: penetration at home, renders from the station's cameras |
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
| Stock DROID open smoke | `cd $XEMB && $PY scripts/datagen/run_pipeline.py --task_type open --scene_dataset src_kitchen_twin --robot droid --planner kino --plan_mode mpc --samples_per_house 2 --num_workers 1 --seed 0 --output_root $MLSPACES_ARTIFACT_ROOT/smoke --run_name droid_open_stock` | **0 of 12 opens as is.** xemb's arm rate limiter, with its default `arm_velocity_limit_reference: measured_qpos`, clips each command to 0.066 rad from the measured joints; the arm then moves at about 0.47 rad/s while the approach is timed for 1 rad/s, so every grasp closed 8-36 cm short. Gravity compensation does not change it. With `--policy_config_json '{"arm_velocity_limit_reference": "previous_command", "mpc_time_cap": "replans_v1"}'`: every grasp holds (TCP error 0.4 mm) and the first 6 of 12 episodes opened (fridge doors, cabinets, a drawer; run `droid_open_prevcmd_replans_n12`). `replans_v1` replaces the wall-clock MPC budget, which ran out on 5 of 7 openings with other simulations loading the Mac |
| Station | `$PY scripts/station_check.py --out data/station_check` | both station poses collision-free at home; the ZED 2i view shows faucet, sinks, range, drawer stacks and dishwasher, the landmarks the record's overlay was checked on |

## Stations

| Key | Arm root (x, y, z, yaw) | Source |
|---|---|---|
| `droid_20261109` (default) | 6.5530, -2.2444, 0.867, 85.95° | `droid_panda_20261109`: ICP of ZED 2i + D435 depth against the kitchen-branch twin |
| `droid_real_v1` | 6.7159, -2.4872, 0.862, 93.57° | main's `SWEEP_BASES["droid_real_v1"]` (2026-10-29 episodes, edge chamfer) |

The two differ by about 30 cm and 7.6°. Main's sweep and population code still uses `droid_real_v1`.

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
