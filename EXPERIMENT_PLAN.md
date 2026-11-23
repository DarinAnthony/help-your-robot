# Experiment plan: when should a robot ask for help?

Status: 2026-11-15. This plan turns the status update's "main approach" into an experiment that can be run and scored. It refines
[`VALUE_PLAN.md`](VALUE_PLAN.md) (the value function) and [`PLAN.md`](PLAN.md) §4 (the decision layer) and takes
precedence over them on the points below: the evaluation protocol, the data the experiment needs, and the baselines.

## 1. Question and hypotheses

**Question.** A robot runs its own manipulation policy and, at each decision point, can **attempt**, **adjust** or
**ask** for help. Does planning over a belief about hidden difficulty, with a learned value function as the prior, decide
better than a value function alone?

- **H1 (predictor).** A value function V trained on rollouts, with images, proprioception and attempt history as inputs, is
  calibrated in distribution (ECE ≤ 0.05). On held-out objects and drawers it ranks episodes better than
  object-frequency and retry-count predictors (AUROC).
- **H2 (belief planning).** At the same help cost, the belief-state planner earns more reward than the best decision rule
  that uses V alone. The gap is largest where an attempt's outcome is informative (pose errors, weak grip, the
  hard drawer), and near zero where the first decision settles the episode (nominal shelf objects).
- **H3 (transfer).** On held-out objects and a held-out drawer, the planner loses less reward than V alone. The
  observation model, P(outcome | difficulty), transfers better than an image-based prior does.

H2 is the primary comparison: the belief planner against the tuned V-with-history threshold, at help cost 3, on the
pooled test set. Everything else is secondary and is reported as such.

## 2. The decision problem

| | Shelf pick-and-place (DROID) | Drawer opening (TidyBot) |
|---|---|---|
| Hidden state | difficulty d ∈ {easy, recoverable, infeasible} | the same |
| Decision points | before attempts 1, 2, 3 | before pulls 1, 2, 3 |
| Actions | **attempt** (the nominal grasp), **adjust** (re-approach swung 15–30°, grasp offset), **ask** | **pull** (re-plan grasp and pull from the current state), **ask** |
| Observations after an attempt | outcome ∈ {success, unreachable, grasp miss, slip, place miss, off shelf}, failure phase, object pose, the decision-point images | drawer travel, hold lost or kept, base travel, arm torques, images |
| Terminal | success, off shelf (object knocked down), place miss, or ask | success or ask |

Difficulty is **defined from data, the same way for both tasks**. It is never hand-assigned and the help policy never
sees it:
- **easy:** the first attempt succeeds;
- **recoverable:** the first attempt fails and some continuation within 3 attempts succeeds;
- **infeasible:** no continuation succeeds.

The shelf's counterfactual trees (§4) make this exact for every start state. For the drawer it comes from the logged
re-pulls.

**Reward.**
- +10 for autonomous success.
- −c_help for asking. The human then finishes the task and the episode ends.
- −c_fail for each failed attempt.
- −5 extra for knocking the object off the shelf. Help is still needed afterwards.
- After the 3rd failure the robot must ask, and pays −c_help.

Default c_help = 3 and c_fail = 0.5, as in the proposal.

With those defaults, asking only pays off on near-hopeless episodes: −3 against −4.5 for three failures. We therefore
**sweep both costs**, c_help ∈ {1, 2, 3, 5, 8} and c_fail ∈ {0.5, 1, 2}. The results are reported as success-vs-help
curves and as regret heatmaps, not at a single point.

## 3. Conditions

**Shelf.** 11 measured kitchen objects × 4 expert modes, with every start state paired across modes (seed0 20000).
- Modes:
  - `nominal`
  - `pose_error`: object-position error σ 1.5 cm
  - `place_error`: target error σ 4 cm
  - `weak_grip`: gripper force × 0.05
- Training objects: tomato, egg, drumstick, wooden ball, banana, waffle, sponge, duster, tape.
- Held out:
  - **orange**, like the training set: it should mostly be attempted;
  - **metal spoon**, thin and flat: it should be asked about early.

**Drawer.** Every pull is planned on the drawer the planner believes in, with nominal physics. The real drawer acts only
once the pull starts (`hidden_drawer`), and a failed pull is re-pulled up to twice (`HELP_REPULLS=2`).

| Condition | Real drawer / fault | Expected class | Split |
|---|---|---|---|
| nominal | xemb's slide friction U(1, 2) N | easy | train / cal / test |
| transient stop | nominal; the first pull's base stops at 30% | recoverable | train / cal / test |
| weak grip | nominal; gripper force × 0.02 from the pull | mixed (4/20 open; after a slip the planner's grasp check aborts, so no re-pull) | train / cal / test |
| hard | slide friction 6.5 N (35% first-pull success, single pull) | recoverable / infeasible | train / cal / test |
| stuck | slide friction 20 N (needs ~60 N) | infeasible | train / cal / test |
| locked | latch at 8 mm, holds 200 N | infeasible | train / cal / test |
| **medium (held out)** | slide friction 4.5 N | mostly easy, some recoverable (22/30 first pulls open; 30/30 within 3) | **test only** |

Drawer travel separates the conditions after a single failed pull:
- locked: 8 mm
- stuck: 19–20 mm
- failed hard pull: 34 mm
- success: about 30 cm

The decision therefore turns on how many pulls to spend on a drawer that might open. Telling the conditions apart is not
the hard part.

## 4. Evaluation protocol: exact offline scoring on the same episodes

Every method is scored on the same test episodes, so comparisons are **paired**.

- **Shelf: counterfactual action trees.** From each start state, the simulator snapshots the full physics state
  (`mj_getState`, including warm start and controls) at every decision point and plays out both **attempt** and
  **adjust**, to a depth of 3 attempts. Each branch uses its own random stream, and decision-point images are rendered
  at every node (`ShelfPnP.run_tree`). Asking is possible at every node.
  - Any policy over {attempt, adjust, ask} therefore has an **exact** reward on each start, with no importance weights
    and no model of the simulator.
  - A tree has at most 14 attempts. The trees reuse the rendered episodes' seeds, so each tree shares its start state
    with a full-rate video episode.
- **Drawer: logged re-pulls.** Each episode is one path of up to 3 pulls. A policy over {pull, ask} only chooses where to
  stop along that path, so its reward on the episode is also exact.
- **Splits.**
  - Seeds of the in-distribution conditions are split 60 / 15 / 25 into train / calibration / test, by start state.
  - The held-out objects (orange, metal spoon) and the held-out drawer (medium) appear only in test.
  - Calibration (Platt scaling) and **every tunable threshold** (τ, k) are fit on the calibration split, separately
    for each cost setting. No method sees test.
- **Statistics.**
  - Paired bootstrap over test start states (10,000 resamples), with 95% CIs on reward differences, overall and per
    condition.
  - For the shelf, there are about 300 in-distribution test starts plus 230 held-out starts. With a paired SD of about 2,
    the standard error is about 0.12 reward per episode, which is enough to resolve a 0.25 difference.
- **Closed-loop check.** The best two methods and two baselines are run live in the simulator on 40 fresh shelf starts
  and 20 fresh drawer episodes. This confirms that the offline scores match the live ones.

## 5. Methods

**Value functions** (VALUE_PLAN.md §3). V_k = P(eventual success | state and history at decision k, the expert keeps
going).
- **V0:** logistic regression on hand features: object size and grasp-width margin, reach margin, IK residual, attempt
  index, last outcome, and drawer travel.
- **V1:** a 5-member ensemble on frozen image embeddings (DINOv2 or SigLIP) plus proprioception and history, then
  Platt-scaled.
- A **difficulty head** (3-way softmax over d) gives the planner's prior.

**Decision methods.** Each one gets its own best threshold or k, tuned on the calibration split.

| # | Method | What it isolates |
|---|---|---|
| B1 | Never ask (always continue; the expert's default attempt-then-adjust) | lower bound on help |
| B2 | Always ask | lower bound on autonomy |
| B3 | Ask after k failures (k = 1, 2) | the standard heuristic |
| B4 | Random asking at the matched help rate | is the timing of asks informative at all? |
| B5 | Threshold on V_0 only (decide once, at the start) | a one-shot success predictor |
| B6 | **Threshold on V_k with history** (re-scored at every decision) | **the value function alone**: the main competitor |
| B7 | Myopic expected utility on V_k (continue iff 10·V_k − c_fail·(1 − V_k) > −c_help; no tuning) | cost-aware V without lookahead |
| B8 | QMDP on the same belief | a planner that ignores the value of information |
| B9 | Fitted Q-iteration over decision points (same features, offline on the training trees) | a learned sequential policy without an explicit belief |
| **Ours** | **Belief POMDP**: prior from the difficulty head, Bayes updates with P(outcome, features \| d, action), exact finite-horizon value iteration (horizon 3) | belief-state planning |
| O1 | Difficulty oracle: the POMDP with d known | value of knowing the hidden state |
| O2 | Clairvoyant: the best path in each tree | upper bound for the regret |

**Ablations of Ours**, each answering part of the feedback question:
1. **Lookahead:** Ours against B7, which uses the same beliefs.
2. **Structured Bayes updates against learned history:** Ours against B6.
3. **Prior:** a V prior against a uniform or object-frequency prior.
4. **Outcome type:** failure categories against success/failure only.
5. **Adjust:** with adjust against attempt only (shelf).

**Implemented (2026-11-23):** `helpsim/help/` (trees, exact scoring, features, the predictors, every method above except
the V1 image ensemble) and `scripts/help_eval.py`, which writes `reports/help_eval/<family>/summary.md`. The same code
scores pi0.5's continue/reset trees (`--family pi05`). First run on the expert trees (`reports/help_eval/expert/`):
Ours beats B6 at low c_fail (CI excludes 0 in 6 of 15 cost settings; never significantly worse), QMDP matches Ours (no
value of information on these trees), and the gain comes from the start-state prior and the adjust action, not from
failure types (ablations A1-A4). Drawer re-pulls (`reports/help_eval/drawer/`, observations refined by the drawer's
travel bin): every method is within 0.4 reward of the clairvoyant and Ours ties B6 (−0.03); 67 test episodes, 30 of
them the held-out medium drawer, are too few to separate methods there.

## 6. Metrics and outputs

**Predictor metrics** (H1): AUROC, Brier, NLL and ECE with reliability diagrams for V0 and V1, at decision points 0, 1
and 2. They are reported per split: in-distribution, orange, metal spoon, each drawer condition, and medium.

**Decision metrics** (H2, H3):
- mean reward with a 95% CI;
- autonomous success rate;
- help rate;
- failed attempts per episode;
- off-shelf rate;
- ask precision and recall against the infeasible label;
- regret to O2.

Figures:
1. success-vs-help curves (one per task) as c_help sweeps;
2. a regret heatmap over (c_help, c_fail) for Ours − B6;
3. the per-condition reward gap with CIs;
4. the belief trajectories of example episodes, with decision-point frames.

**What would falsify H2:** Ours − B6 has a CI that contains 0 at every cost setting in the recoverable conditions. That
is a reportable result, and it would mean that a value function trained with attempt history already captures what
the belief adds.

## 7. Data: what exists, what is collected, and why

| Data | Episodes / starts | Status | Used for |
|---|---|---|---|
| Shelf 20261113, nominal, seed0 1000 | 275 | done, on Drive | V training (extra starts) |
| Shelf 20261114, 4 modes, seed0 20000, rendered | 440 + 275 + 275 + 275 | done, on Drive | V training, images |
| **Shelf action trees**, same 1,265 starts | 12,892 attempts; easy 374, recoverable 252, infeasible 639 | done, on Drive (`shelf_pnp_droid_20261114_tree`) | the exact evaluation; the observation model for attempt and adjust; difficulty labels |
| Drawer 20261113b, single pull, planner-aware | 50 | done, on Drive | first-pull V only |
| Drawer 20261114, single pull, hidden faults | nominal 30, hard 20, stuck 20, locked 20 | done, on Drive | first-pull V and observation model |
| **Drawer re-pull runs** (`HELP_REPULLS=2`, `tidybot_open_rp_*_20261115`, seeds 2000–2615) | nominal 20/20, transient stop 20/20, weak grip 4/20, hard 32/40, stuck 0/25, locked 0/25, medium 30/30 opened | done, on Drive (trajectories; sessions exporting) | sequential drawer decisions; the drawer evaluation |

More drawer data **is** necessary: until now no drawer episode contained a second attempt. Without that, the drawer can
only test a one-shot decision, so neither the belief update nor H2 can be tested on it. The re-pull runs give real
post-failure states, with the drawer left where the failed pull left it. The single-pull runs stay useful as first-pull
data.

The shelf needs no more rendered episodes. It needs the trees, which give the counterfactual attempt/adjust outcomes
for the same starts.

## 8. Timeline

| Dates | Work |
|---|---|
| Nov 15–17 | Drawer re-pull runs and shelf trees. Session exports, Drive upload, README. |
| Nov 17–21 | Feature extraction (H5 streams: drawer travel, torques; tree states; image embeddings). V0, V1 and the difficulty head. The calibration report (H1). |
| Nov 22–26 | Observation models. The belief POMDP, B1–B9, the oracles, and the offline tree scorer with bootstrap. |
| Nov 27–Dec 3 | Cost sweeps, ablations, held-out conditions. The closed-loop check. |
| Dec 4 onward | Figures and the final report. Stretch: a π0.5 policy in place of the scripted expert (GPU). |

## 9. Threats to validity

- **A scripted expert, not a learned policy.** The difficulty structure comes from that expert, so the claims are about
  help-seeking on top of a fixed skill.
- **Drawer re-pulls.** The planner re-plans on the believed drawer each time. While the hand approaches for a re-pull,
  the drawer has nominal physics, until the pull starts.
- **Weak grip ends itself.** After a slip the planner's own grasp check rejects every grasp, so weak-grip episodes
  stop after one pull instead of offering a re-pull. On the drawer, weak grip is effectively a one-shot condition
  where the planner detects the fault; the sequential test rests on transient stop, hard, stuck and locked.
- **The held-out drawer is easy-leaning.** `medium` opened in 30/30 (8 needed a re-pull), so it tests over-asking on
  mostly easy drawers more than recovery from hard ones.
- **Outcome-defined difficulty.** The labels depend on the attempt budget (3). Reporting at budgets of 2 and 3 shows
  that the conclusions do not hinge on it.
- **The kitchen's object set has no cups.** Generalization is tested across shelf objects and drawer frictions, not
  object categories.
- **Simulation only.** The physics is identified, but no real-robot evaluation is planned.
