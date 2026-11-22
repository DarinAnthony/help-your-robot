# Task specification: pi0.5 base policy + help decisions

Draft, 2026-11-21. Settles what pi0.5 learns, how it is run in sim, and what the help model decides, so data
conversion, training and evaluation all build on the same definitions. `EXPERIMENT_PLAN.md` keeps the hypotheses,
baselines and statistics; where this file and the plan differ, this file is newer.

**Decided (2026-11-21):** both tasks; asking for help = oracle success; the help model decides at every attempt
boundary; difficulty comes from pi0.5's own failures plus injected faults.

## 1. The system

```
 instruction ──► pi0.5 (LoRA fine-tune) ──► actions, closed loop at 15 Hz
                     │
   sim ground truth: attempt ends (success / failure type / timeout)
                     │
                     ▼
 help model ── observation: images, proprioception, attempt index, previous outcomes, pi0.5's own spread
            └► continue  |  reset and retry  |  ask
```

- **Base policy:** pi0.5 from the public base checkpoint, LoRA fine-tuned on successful scripted-expert episodes.
- **Failure detection:** sim ground truth (§4). There is no learned failure classifier.
- **Help model:** decides before every attempt (§5). It is the thing under study.

## 2. Task A: shelf pick and place (DROID Franka)

| | |
|---|---|
| Scene | SRC kitchen twin; DROID at the cart station (`helpsim/tasks/shelf_pnp.py`) |
| Instruction | `put the {object} on the shelf` |
| Objects | train: wooden ball, tomato, egg, drumstick, waffle, sponge, duster, banana, tape. Held out: orange (in distribution), metal spoon (hard). Held-out objects never appear in pi0.5's training data. |
| Start | object pose uniform in the counter region, yaw uniform; shelf target sampled (`ShelfPnP.reset`) |
| pi0.5 observation | `cam_cabinet` (exterior) + `wrist_camera`, 224×224; state: 7 arm joints + gripper |
| pi0.5 action | absolute arm joint targets (7) + gripper command (1), 15 Hz; chunk of 32, executing the first 8 |
| Success | object at rest on the shelf within 5 cm of the target (`PLACE_TOL_M`) |
| Attempt limit | 20 s (300 ticks; expert attempts take 10–12 s); 3 attempts per episode |

**Training data.** Every successful expert episode on the 9 training objects, including those that succeeded on a
retry (they show recovery), from all four expert modes of 2026-11-13/14: 435 (tape has none: the expert never
succeeds with it). The rendered videos are 5 Hz, but
pi0.5 needs a frame on every 15 Hz step. Episodes are deterministic in their seed, so the successes are re-run with
`--render-every 1 --rerender-from <old run>` (`scripts/collect_shelf_pnp.py`); every re-run is checked to reproduce
its outcome, and the first ones matched the originals' states exactly. **More data:** 1,050 new nominal episodes on the
7 objects the expert can do plus 150 weak-grip banana episodes (the mode where the banana succeeds), about 600 more
successes, for about 1,000 demos (`data/shelf_pnp/pi05_demos/`). 5% of them, chosen at random, are held out
(`help/shelf_pi05_val`) for the open-loop check.

## 3. Task B: drawer opening (TidyBot)

| | |
|---|---|
| Scene | SRC kitchen twin, demo drawer `glass_counter__stack_b_3_front`; TidyBot (xArm7 on a mobile base) |
| Instruction | `open the drawer` |
| Drawers | nominal (train); hard, stuck, locked, transient stop, weak grip (fault conditions, evaluation only); medium (held out, test only) |
| pi0.5 observation | mast camera + wrist camera, rendered as in collection, 224×224; state: 7 arm joints + gripper + base pose (x, y, θ) |
| pi0.5 action | arm joint targets (7) + gripper (1) + base command (3), xemb's ~15 Hz policy step; chunk of 32 |
| Success | xemb's `judge_success` (drawer open) |
| Attempt limit | 25 s per pull; 3 pulls per episode |

**Training data.** Successful **nominal** episodes only, so every fault drawer stays unseen by pi0.5: 70 now
(20261113b, 20261114, the re-pull nominal run) plus 300 new, 5% held out as `help/drawer_pi05_val` (`scripts/collect_drawer_open.sh`, after the shelf
collection: the two do not fit in memory together). `tidybot_open_20261113` is left out (its preview cameras predate the
hooks).

The training cameras are the sim cameras rendered during collection, the same ones rendered at evaluation. The
calibrated-camera sessions matter only for real-robot transfer, which is out of scope.

## 4. Attempt boundaries (sim ground truth)

The scripted expert's attempts were discrete. pi0.5 runs closed loop, so an attempt ends at the first of:

| Event | Shelf | Drawer |
|---|---|---|
| **success** | object on the shelf within 5 cm, at rest for 0.5 s | drawer judged open |
| **grasp miss** | gripper closed with no object contact, or the object not lifted 3 cm within 2 s of closing | gripper closed without the handle |
| **slip / drop** | the object leaves the gripper while carried and comes to rest off the shelf | hold lost during the pull |
| **stall** | — | drawer travel under 5 mm over 2 s while holding |
| **off shelf** (terminal) | the object fell off the counter or shelf | — |
| **place miss** | released on the shelf more than 5 cm from the target | — |
| **timeout** | no other event within the attempt limit | the same |

The outcome type is part of the help model's observation.

## 5. The help decision

- **When:** before each attempt k = 1, 2, 3: at the start, and after each failed attempt.
- **Actions:**
  - **continue:** pi0.5 resumes from the current state;
  - **reset and retry:** arm to home, gripper open (and the base back to its start, on the drawer), then pi0.5;
  - **ask:** the episode ends in success (oracle help).
- **Reward:**
  - +10 for autonomous success;
  - −c_help for asking (the oracle then finishes, with no +10);
  - −c_fail per failed attempt;
  - −5 extra for knocking the object off the shelf;
  - after the third failure the robot must ask.

  Defaults c_help = 3 and c_fail = 0.5; both are swept as in `EXPERIMENT_PLAN.md` §2.
- **Help model inputs:**
  - images at the decision point (frozen encoder features, or pi0.5's own vision embedding);
  - proprioception, the attempt index and the previous outcomes;
  - **pi0.5's uncertainty:** the spread of N sampled action chunks from the same observation. Flow sampling makes
    this free at inference, and it is a direct answer to "how uncertain is the policy about this attempt?".

## 6. Difficulty and the evaluation trees

Difficulty is defined per start from outcomes, as before (easy / recoverable / infeasible within 3 attempts).

- **Natural:** objects and starts where pi0.5 itself fails.
- **Injected:**
  - shelf: weak grip and slippery pads (`grip_scale`, `grip_friction`). The expert's pose error and place error are
    noise in the expert's perception and targets, which pi0.5 does not have, so they are not faults for it;
  - drawer: hard, stuck, locked, transient stop, weak grip.

The exact offline evaluation needs, per start, what every decision would have led to. As with the expert trees, the
sim state is saved at each decision point and both continue and reset-and-retry are run from it: at most 14 pi0.5
attempts per start. pi0.5 samples, so each branch is one draw. For the calibration split, 3 draws per branch give
outcome probabilities.

## 7. Where things run

**Stack:** upstream openpi (`Physical-Intelligence/openpi`, pinned in `helpsim/openpi_help.py`) with LeRobot datasets.
xemb's vendored openpi (`pi05/`) imports xemb's private training package and its RLDS contracts, so it cannot run on
Colab alone. Upstream needs no edits: `helpsim/openpi_help.py` adds the two configs (input/output transforms, arm and
base deltas within each chunk with the gripper absolute, pi0.5 LoRA from `pi05_base`, no EMA).

| Piece | File |
|---|---|
| Expert episodes → LeRobot (180×320 H.264 video, absolute actions, observation k with the command of step k+1), 95% train / 5% `_val` | `scripts/to_lerobot.py` |
| openpi configs | `helpsim/openpi_help.py` |
| Norm stats, training, open-loop check on `_val`, policy server | `scripts/openpi_run.py` |
| Seconds per update and the run size, from the training log | `scripts/step_timer.py` |
| Colab notebook | `notebooks/pi05_train_colab.ipynb` |

mikulrai/openpi was also considered, for its LoRA-only checkpoints and validation loss. It adds a fourth image slot
that is always embedded, even when masked, so each update costs ~25–30% more on these two-camera tasks. Upstream is
used instead: the held-out `_val` dataset gives the validation, and the notebook keeps checkpoints on local disk and
mirrors them to Drive every 5 minutes. A full pi0.5 checkpoint is ~7 GB, so Drive holds the latest plus every 5000th.

Checked on the Mac (openpi on CPU): both tasks' test datasets load through openpi's data pipeline and give norm stats;
the shelf config trains, checkpoints, resumes and runs the open-loop check end to end with dummy-size Gemma weights.

- **Training:** Colab, 200 compute units (A100 class), data and checkpoints on Drive.
- **Evaluation (recommended):** the sim stays on the Mac, which already has every asset. pi0.5 runs on Colab as
  openpi's websocket policy server behind a tunnel, so the GPU is used only for inference. Latency (~0.1–0.3 s per
  chunk) does not matter because the sim is not real time. The alternative is moving xemb and the twin's assets to
  Colab, with EGL rendering.

| Step | Rough cost (to be measured in the smoke run) |
|---|---|
| Smoke train + norm stats | 5–10 units |
| Shelf LoRA, 10k updates | ~60–90 units |
| Drawer LoRA, 10k updates | ~60–90 units |
| Policy serving for the evaluation trees | ~20–40 units |

That is close to 200, so **each run is sized from the smoke run's measured seconds per update**. If it does not
fit, the drawer run gets fewer updates first.

## 8. Timeline (report from Dec 4)

| Dates | Work |
|---|---|
| Nov 21–23 | Re-render shelf successes at 15 Hz; collect more nominal shelf and drawer episodes; converters to RLDS; upload |
| Nov 23–24 | Colab environment, smoke train, norm stats; size the runs |
| Nov 24–27 | Shelf LoRA, then drawer LoRA |
| Nov 26–29 | Policy server; attempt detectors (§4); rollouts and evaluation trees with pi0.5 |
| Nov 30–Dec 3 | Help model and baselines on the pi0.5 trees; the cost sweeps |

**Fallback.** The scripted-expert trees and re-pull runs (`EXPERIMENT_PLAN.md`, data on Drive) support the whole
help-decision study without pi0.5. That track can run in parallel on the Mac, so a late or weak pi0.5 still leaves a
complete project, with pi0.5 as the extension.

## 9. Open decisions

1. **One joint model or two:** two, shelf first (decided 2026-11-21).
2. **Action representation:** joint position targets as collected, trained as deltas from each chunk's first state
   (openpi's convention for absolute-action data); the policy returns absolute targets (decided 2026-11-21).
3. **The tunnel** for the policy server: a Cloudflare quick tunnel (no account) in the notebook. It exposes the
   policy server to anyone with the URL while it runs.
4. **A time cost** in the reward beyond c_fail: none by default.
