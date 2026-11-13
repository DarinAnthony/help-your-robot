# Task-completion value function: plan

Status: design draft, 2026-11-13. Builds on [`uncertainty-methods-report.md`](uncertainty-methods-report.md) and
[`PLAN.md`](PLAN.md). Nothing here is trained yet.

## 1. The question the value function answers

At a decision point *t* (before an attempt, or after a failed one), estimate

**V(s_t) = P(the task succeeds if the robot keeps executing its own policy from s_t)**

together with **how much to trust that estimate**. The help policy then compares the expected value of attempting
or retrying with the value of asking. This is the "capability / outcome uncertainty" the report identifies as the
open problem (§B, §C). Semantic ambiguity (KnowNo-style) is a separate, solved component and stays out of v1.

For a binary terminal reward and no discount, the Monte Carlo value of a state equals the success probability under
the data-collecting policy. So the value function is a **time-indexed success classifier**, and every
calibration tool for classifiers applies to it (Brier, NLL, ECE, Platt scaling).

## 2. Data this needs (being collected now)

| Task | Robot | Training objects / conditions | Held out |
|---|---|---|---|
| Shelf pick-and-place, `upper_return` middle shelf (z 1.51 m) | identified DROID, door-session cart pose | the kitchen's measured objects except two | **A: in distribution**: an object like the training set (tests generalization, should mostly *attempt*). **B: hard to grasp on purpose**: should be predicted low and trigger *ask* or *retry* |
| Drawer opening, `glass_counter::stack_b_3` | identified TidyBot at its demo base pose | measured drawer physics plus the existing friction and damping bands | **Evaluation drawer made hard on purpose**: high friction, or locked (joint range clamped) |

Every episode keeps its **failures** (`--debug_failure_rate 1`), with labels that say why it failed (missed grasp,
slipped, unreachable, drawer did not move). Retries come from re-running the planner after a failed attempt, from the
state the failure left. That makes post-failure states real training points: V must update after a slip.

Splits: train / calibration / test by episode seed. Held-out objects A and B, and the hard drawer, appear **only** in
test. Calibration uses in-distribution episodes only, as the report insists (VLAConf: calibrate on a few labeled
in-distribution rollouts).

## 3. Model options, from cheapest to richest

| | Model | Uncertainty it gives | Why / why not |
|---|---|---|---|
| **V0** | Logistic regression on hand features (object size and grasp-width margin, shelf depth, reach margin, IK residual, retry count, last-attempt outcome) | P(success) only | Baseline. The report notes a plain logistic probe calibrates about as well as fancier ones (VLAConf NLL 0.253 vs. 0.255). |
| **V1 (recommended first)** | **Deep ensemble** (5 MLPs) on frozen image embeddings (wrist + exterior; SigLIP from π0.5, or DINOv2) + proprioception + attempt history, trained with binary cross-entropy against the episode outcome at every step, then **Platt-scaled** | Mean = P(success) (aleatoric). Member disagreement = epistemic (should be high on held-out object B and the hard drawer) | Cheap to train on a Mac-sized dataset. Separates "I'll likely fail" from "I don't know". The POMDP can treat the two differently. |
| V2 | Same features, **TD / IQL value** over high-level steps (ATTEMPT, ADJUST, INSPECT, ASK), trained offline on the attempt dataset | Q(s, a) for each action directly | Closer to Jiang et al.'s offline-RL precedent. Needs more data and is harder to calibrate. Compare against V1 + POMDP rather than replacing it. |
| V3 | Probe on π0.5 internals (SAFE / VLAConf style), once π0.5 is fine-tuned | P(success), discrimination from policy features | Only after a learned skill exists (PLAN.md M4). The report shows SAFE drops on unseen tasks (86.8 → 64 on a real Franka), so expect V3 ≥ V1 in distribution, not necessarily on object B. |

Not recommended as the main signal: raw action or token entropy (report: AUROC 0.468 on long-horizon tasks), and VLM
success detectors for the drawer (FailBench: ≤ 0.60 on contact-heavy tasks; a failed pull looks like nothing
happened). Drawer features should include **force and displacement** (the TidyBot has a wrist F/T sensor, and the
sim has joint torques): commanded pull versus drawer travel.

## 4. From V to a decision

The value function is an input, not the policy. Following the report's §4:

1. **Belief state.** A discrete latent per episode, {easy, recoverable, infeasible}, with the prior from the
   calibrated V (and the ensemble spread as a confidence on that prior). After each attempt, Bayes-update with an
   **empirical observation model**: P(outcome, V-after | latent), fitted on the training episodes.
2. **Solver.** Finite-horizon belief value iteration (3 latents, 4 actions, horizon ≤ 6 is tiny), plus POMCP as a
   check. **Not QMDP for the main policy**: it never values pure information-gathering and so never asks (Rosenthal &
   Veloso). QMDP stays as a baseline that shows this failure.
3. **Reward.** +10 success, −3 help, −0.5 failed attempt, −0.02 per step, −5 unsafe. Sweep the help cost.

Where it should beat a threshold:
- **Object B** (low P(success), *low* uncertainty): ask immediately. A threshold on uncertainty gets this wrong.
- **Shelf back edge or object A** (middling P, cheap attempt): try once, then update. A fixed P-threshold asks too early.
- **Hard drawer**: one torque-capped probe pull, then ask (the force observation resolves "stiff" vs. "locked").

## 5. Baselines and metrics

Baselines (same skills, same episodes): never ask; always ask; random asking at the matched help rate; retry-k;
threshold on calibrated P(success); threshold on ensemble uncertainty; myopic expected utility on the *same* beliefs
(isolates the gain from sequential planning, the report's open question 3); QMDP.

Metrics:
- **Predictor quality:** AUROC, Brier, NLL and ECE of V, reported separately for in-distribution, object A,
  object B and the hard drawer.
- **Decision quality:** success vs. help-rate curves, pAUC at low help budgets, and total reward under the cost model.

## 6. Steps

1. **Data (this week, on the Mac):** shelf PnP with the DROID (train objects + A + B) and drawer opens with the
   TidyBot, failures kept, success and failure reasons labelled. Upload each run with `scripts/push_to_drive.sh`.
2. **V0 and V1:** features extracted from the H5s; train; calibrate; report the predictor metrics table.
3. **Observation model + belief planner + baselines** on recorded attempts (offline replay of attempt outcomes).
4. **Hard drawer + object B evaluation in the full sim** with the policy in the loop.
5. Optional: V2 (offline RL) and V3 (π0.5 probe) once a π0.5 checkpoint exists (cluster GPU).

## 7. Open choices

- **Image encoder:** π0.5's SigLIP (consistent with V3 later) or DINOv2 (no openpi environment needed on the Mac).
- **Decision granularity:** per skill call (recommended) or per fixed time window.
- **Can V alone be the observation model?** Or should the attempt outcome (slip vs. miss vs. unreachable) enter the
  belief separately? Fit both and compare log-likelihood on the calibration split.
