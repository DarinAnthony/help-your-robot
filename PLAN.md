# help! — When should a robot ask for help? (CS 238 project plan)

Status: planning draft, 2026-11-13. Nothing is implemented yet. This plan replaces the pasted spec's
ManiSkill 3 + simulated Franka setup with the **SRC kitchen MuJoCo twin** and the **identified DROID station**
from `~/research/xemb-vla`. The research framing (a belief-based POMDP over ATTEMPT / ADJUST / INSPECT /
ASK_HELP, compared with threshold and myopic baselines) stays the same.

## 1. What already exists in xemb-vla (and what we reuse)

| Need | Already in `~/research/xemb-vla` | Notes |
|---|---|---|
| Kitchen scene | `envs/src_kitchen/` twin: 61 articulated parts, 10 placement surfaces, measured shelves inside cabinets, real lighting | **Generated, not committed.** It must be installed from HomeBody `c09fcf39` (`scripts.analysis.homebody_kitchen.twin --install`) or linked from a staged copy (`scripts/setup/src_kitchen_twin_assets.py link --version v1`). |
| Robot at the real station | `TWIN_ROBOTS` row `droid` (stock `franka_droid`: FR3 + Robotiq 2F-85) at the identified arm height 0.867 m (`task_set.DEPLOYMENT_STATIONS`) | Base pose `SWEEP_BASES["droid_real_v1"] = (6.7159, -2.4872, 0.862, yaw 93.572°)` in `house_tasks.py`, ±3 cm / ±1°. |
| Real cameras | `camera_configs.droid_record_cameras("droid_panda_20261103")` (profile `droid_record_v1`): wrist ZED Mini + overhead ZED 2i | Pinholes matched to the real lenses. |
| Identified dynamics | `molmo_spaces/robots/measured/droid_panda.py` + `droid_dynamics.py`: Panda torque plant at 1 kHz, libfranka 100 Hz filter, lumped 0.97 kg tool; 5 mm / 1° closed-loop tracking | **Standalone builder**, not wired into `run_pipeline.py`. Gives real joint torque limits, which we need for the "infeasible drawer" condition. |
| Drawer / door physics | Measured soft-close zones (11→28 N), 0.2–0.4 N slide friction, upper-door hinge 2.4 N·m (`reference/src_kitchen_measurements.json`, `twin.JOINT_MODELS`) | We add the hidden variable (friction, lock) on top of these. |
| Planners / scripted expert | `run_pipeline.py --robot droid --planner kino --plan_mode mpc` (open/close), pick-and-place sampler `SrcKitchenTwinPickAndPlaceTaskSampler`, RRT planner `--use_motion_planner` | This is our "IK/scripted expert" for Milestone 1. |
| Objects | `envs/src_kitchen/objects/`: banana, egg, orange, tomato, waffle, drumstick, plate, sponge, tape, duster, spoons, wooden ball | **No cups.** Cups/mugs come from THOR (`...PickAndPlaceThorDataGenConfig`, no download) or Objaverse, which conveniently makes them a clean held-out category. |
| Learned policy | `pi05_integration/`, `openpi_integration/`, `octo_integration/`, `molmo_spaces/evaluation/` with explicit-state twin eval | π0.5 is the natural VLA (action sampling available → action-disagreement uncertainty). |

**Local machine reality (checked):** this Mac has no `mlspaces` env, no `mujoco`, no HomeBody clone and no
installed twin. Simulation work runs either on the cluster (`README_CLUSTER.md`) or on watney. See §7.

## 2. Mapping the two tasks onto the kitchen + DROID station

The DROID is a **fixed-base station** at the island's south end, so reachability is set by the station, not by
a moving base. That is useful: "infeasible because unreachable" is a real, fixed property of the target.

### Task A — Shelf / cabinet retrieval (held-out cups)

- **Where:** the cabinets whose shelves are measured (`src_kitchen_interiors_measured.json`): uppers at
  1.1985 / 1.5065 / 1.7945 m, base shelves at 12–46 cm, plus island / sink worktops as "easy" surfaces. The real
  DROID already opened `upper_return::door_1` from this station, so the upper return cabinet is near the reach
  boundary — ideal for the "recoverable" and "infeasible" conditions.
- **Instruction:** "Retrieve the [object] from the [cabinet/shelf] and place it in the [tray/sink/target area]."
  Door pre-opened in Milestone 1 (fold door opening in later, as an extra source of failure).
- **Difficulty (ground truth, hidden from the help policy):**
  - *easy* — lower upper shelf / worktop, object upright, front of shelf, nominal friction;
  - *recoverable* — object rotated, partially occluded, pushed back, or low friction; a second grasp or a
    re-approach can succeed;
  - *infeasible* — top shelf (1.79 m) or deep in the cabinet beyond the arm's reach from `droid_real_v1`, or
    blocked by another object.
  Labels are verified by repeated scripted-expert trials (e.g. ≥ 8/10 easy, 1–7/10 recoverable, 0/10 infeasible
  with all grasp strategies) — never assigned only by hand.
- **Object split:** train = twin objects + THOR non-cup categories (bottles, cans, boxes, cartons, the fake foods);
  held out = THOR Cup / Mug (with and without handles, several sizes). Keep the 2×2 (seen/unseen × easy/infeasible)
  evaluation grid from the spec.

### Task B — Stuck drawer (hidden mechanics)

- **Where:** the drawer stacks / glass-counter drawers reachable from the station (Milestone 0 checks which). These
  already have measured soft-close catches, so "normal" is calibrated to the real kitchen.
- **Hidden variable per episode:** slide friction (×1 … ×50), breakaway force, a lock (joint range clamped to
  ~0), or an obstruction box inside the drawer.
- **Why the identified Panda matters here:** with the stock position servos the arm can exert effectively
  unbounded force, so nothing is truly "infeasible". The DROID Panda torque plant (`droid_panda.TorqueResponse`)
  has real joint torque limits and a 100 Hz torque filter, so "the drawer requires more force than the robot can
  provide" is physical rather than a scripted cutoff. Plan: run Task B on the identified plant (or at minimum cap
  the stock actuators' `forcerange` to the Panda limits and document it).
- **Observations the help policy gets:** displacement vs. commanded pull, joint torques / `tau_ext`, gripper slip.

The **jar** task from the first spec is dropped: the twin has no jar, and the drawer covers "help despite high
confidence" (a locked drawer).

## 3. Architecture

```
help!/
├── PLAN.md                     # this file
├── README.md                   # setup, tested commands, open problems (written in Milestone 1)
├── helpsim/                    # importable package; depends on xemb-vla via PYTHONPATH / editable install
│   ├── envs/
│   │   ├── kitchen_station.py  # load installed twin + DROID at droid_real_v1 + record cameras
│   │   ├── shelf_retrieval.py  # Task A sampler: object, shelf cell, difficulty knobs, success check
│   │   └── stuck_drawer.py     # Task B: hidden friction / lock / obstruction on a twin drawer
│   ├── skills/                 # execute_skill(obs, instruction, skill_state) -> actions, diagnostics
│   │   ├── scripted.py         # wraps xemb planners (kino / RRT grasp selector)
│   │   └── vla.py              # π0.5 client (later)
│   ├── uncertainty/            # action disagreement, failure predictor, OOD score, calibration
│   ├── help/                   # belief update, POMDP planner, baselines
│   └── logging.py              # per-episode JSON + video, same layout for every policy
├── configs/                    # object splits, difficulty bands, reward table
├── scripts/                    # demo_shelf.py, demo_drawer.py, collect_rollouts.py, evaluate.py
└── data/                       # git-ignored
```

Rules: xemb-vla is used as a library, never copied or edited from here (changes it needs go upstream as their own
PR). The help policy never reads simulator difficulty labels; the env exposes them only to the logger.

## 4. Decision layer (the actual CS 238 contribution)

- **High-level step = one skill call** (approach+grasp, lift, place, pull). The planner never sees per-timestep control.
- **Actions:** `ATTEMPT`, `ADJUST` (new grasp candidate / re-approach / regrip handle), `INSPECT` (wrist-camera
  look from a second pose; a light, torque-capped test pull), `ASK_HELP` (terminal; human completes).
- **Belief:** discrete over {easy, recoverable, infeasible} per episode, initialized from the uncertainty
  features (failure predictor, OOD score, reach margin), updated by Bayes from outcome observations (grasp
  failed / slipped / unreachable / moved < x cm / torque saturated). Observation models are *fitted from scripted
  rollouts*, so the POMDP is grounded in the sim, not hand-tuned.
- **Solver:** exact finite-horizon belief-space value iteration (3 states, ≤ 4 actions, horizon ≤ 6 is tiny),
  then optionally QMDP / POMCP as comparisons, and a learned policy if time allows.
- **Reward:** spec's table (+10 success, −3 help, −0.5 failed attempt, −0.02 per step, −5 unsafe), configurable,
  with a sweep over the help cost.
- **Baselines:** always-attempt, always-ask, ask-after-k, threshold on p(success), myopic expected-reward, ours.
  All share the same episodes and skills.

An important shortcut the spec already endorses: once skill outcome models are fitted from sim rollouts, most
policy comparisons can run on a **fast surrogate simulator** (sample hidden state → sample observations from the
fitted models), with the full MuJoCo kitchen used to fit the models and to validate the final policies.

## 5. Milestones

**M0 — Environment bring-up (first).**
1. Pick the compute host (§7) and create the `mlspaces` env there (`README_MOLMOSPACES.md` → Installation).
2. Install or link the twin; run `pytest tests/test_src_kitchen_twin_env.py`.
3. Run the stock smoke: `run_pipeline.py --task_type open --scene_dataset src_kitchen_twin --robot droid --planner kino --plan_mode mpc --samples_per_house 2 --num_workers 1`.
4. Stand the DROID at `droid_real_v1` with the station height and `droid_record_v1` cameras (`exact_robot_base_pose`), render both cameras, and compare with a real frame.
5. **Reach map from the station:** for every placement surface, shelf cell and drawer, record whether the IK /
   grasp selector finds a solution. This decides which shelves/drawers are easy / boundary / infeasible.

**M1 — Shelf retrieval env with a scripted expert** (the spec's "Milestone 1"): randomized object/shelf cell,
success detector (lifted, placed within r of target, not dropped), failure-type labels, videos + JSON logs,
category split config. Demonstrate success and each failure mode.

**M2 — Stuck drawer env** on the identified Panda plant (or torque-capped servos), three hidden conditions, force
and displacement observations. Moved earlier than in the spec because it needs no learned policy and gives a
complete POMDP testbed quickly.

**M3 — Outcome models + POMDP + baselines** on scripted skills, for both tasks. *This alone is a complete CS 238
project.*

**M4 — Learned skill + uncertainty (stretch):** fine-tune π0.5 on non-cup retrieval demos from M1, then action
disagreement, a SAFE-style failure predictor, and OOD scores; evaluate calibration (AUROC, Brier, ECE) on held-out
cups; swap these features into the belief initialization.

## 6. Risks

- **Twin install** depends on HomeBody `c09fcf39`; an existing staged copy (`src_kitchen_twin_v1`) is safer than reinstalling.
- **Station placement:** the stock DROID's 0.5 × 0.5 m support box at 0.867 m never placed in stove8; use
  `robot_base_footprint_m=(0.3, 0.3)` as the stove waves did, or a fixed `exact_robot_base_pose`.
- **Identified plant not in the generic path:** wiring `droid_panda` into the task samplers is real work; the
  fallback is capping actuator force on the stock DROID.
- **VLA fine-tuning cost** — kept as a stretch goal so the project does not depend on it.
- **Cups in pretraining:** π0.5 has seen cups; we can only claim cups are excluded from *our* fine-tuning.

## 7. Open decisions

1. **Compute host:** cluster (Sherlock / SC per `README_CLUSTER.md`), watney (has the real-robot DROID data but no
   `mlspaces`), or install `mlspaces` on this Mac (MuJoCo works with `MUJOCO_GL=cgl`; slow for datagen).
2. **Drawer dynamics:** identified Panda torque plant (faithful, more integration) vs. stock DROID with capped forces (fast).
3. **Low-level skill for the main results:** scripted planners only (recommended for the deadline), or π0.5.
4. **Deadline / team size**, which decide whether M4 is in scope.
