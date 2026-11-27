# Final-week scope (Nov 27 – Dec 8, 2026)

Draft, 2026-11-27. Locks what the final report claims, which result is the main one, and what we stop doing. Where
this file disagrees with `PLAN.md`, `EXPERIMENT_PLAN.md` or `TASK_SPEC.md`, this file wins for the final week. The
deadline comes from the CS238 Final Project doc's timeline ("Dec 5–8: evaluate … and finish writing the final
report"). **Unknown: the report's format and page limit. Check the course page.**

## 1. The project in three sentences

A robot attempts a manipulation task up to 3 times. Before each attempt it chooses **continue** (try again, or adjust
the approach) or **ask for help**. Reward: +10 for succeeding alone, −3 for asking, −0.5 for each failed attempt. We
compare ways of making that choice.

## 2. What the report claims

What the proposal promised: *"A learned success predictor or value function may estimate the expected return of
continuing autonomously ... We will compare this approach against simple baselines, such as always asking, never
asking, or requesting help after a fixed number of failed attempts."*

| Claim | Comparison | Status (scripted expert, shelf, 488 test starts, default costs) |
|---|---|---|
| **C1 (must have).** A learned predictor beats the simple rules | V threshold (B6) and Ours vs. never ask / always ask / ask after k | **Holds.** Never ask 0.60, ask after 2 0.59, always ask −3.00; B6 1.00, Ours 1.37. Ours − never ask = +0.77 [0.37, 1.17]. |
| **C2 (main question).** Belief planning beats the value function alone | Ours (belief POMDP) vs. B6 | **Borderline.** +0.37 [0.00, 0.75]. The CI excludes 0 in 5 of 15 cost settings; Ours is never significantly worse. |
| **C3 (RL help model, from the Game Plan).** An offline-RL help policy | B9 fitted Q-iteration vs. the rest | Done: 0.93 at default costs. Better than the simple rules, worse than Ours (+0.44 [0.21, 0.67] for Ours). |
| **C4 (secondary).** The pipeline runs on a learned policy | pi0.5 shelf and drawer | Runs end to end. Too few test episodes to rank the methods (drawer: 29 test episodes, every learned method ties never-ask; shelf: 21 starts, 17 of them impossible). |

**How to state C2 honestly:** QMDP ties Ours (1.38 vs 1.37), and "success/failure only" also ties Ours. The gain over
B6 comes from **the start-state difficulty prior plus the adjust action** (ablations A1, A2, A4), not from lookahead and
not from failure types. The headline is therefore "a structured belief (prior + Bayes updates) beats thresholding a
value function", not "planning ahead helps".

## 3. Definitions to use everywhere

- **Simple baselines:** B1 never ask, B2 always ask, B3 ask after k failures (k tuned on the calibration split), B4
  random asking at the same help rate.
- **Main baseline:** **B6**, a threshold on V_k = P(success if the robot keeps going | what it has seen so far),
  recomputed before every attempt. This is the value-function-alone approach the proposal describes.
- **Ours:** the belief POMDP over {easy, recoverable, infeasible}. The prior comes from a difficulty classifier, Bayes
  updates follow each attempt's outcome, and exact value iteration plans over the 3-attempt horizon.
- **RL help model:** B9, fitted Q-iteration on the same features, trained offline on the training trees. This stands
  in for the Game Plan's "train the help model with RL". We will **not** build the CNN/RNN version.
- **Main result** = scripted expert, shelf. Drawer (expert) = a tie, reported in one line. pi0.5 = the C4 demo.

## 4. Day-by-day

| Dates | Work | Done when |
|---|---|---|
| Nov 27–28 | **Trust the numbers.** A team member (not an AI) reads the scoring in `helpsim/help/trees.py` and the policies in `helpsim/help/policies.py` (B1, B2, B3, B6, Ours). Recompute never ask and always ask by hand from 5 trees. Re-run `scripts/help_eval.py --family expert` and check it reproduces `reports/help_eval/expert/summary.md`. | Numbers reproduce; any bug found is fixed and the eval re-run. |
| Nov 29–30 | **Figures.** (1) Success vs. help rate as the help cost sweeps. (2) Ours − B6 heatmap over costs. (3) One example episode: belief and decision at each attempt, with frames. (4) The value function's reliability diagram. Most of these already exist in `reports/help_eval/expert/`; tidy them for the paper. | 4 figures we would put in the report. |
| Dec 1–4 | **Write the report.** Problem and POMDP; tasks and data; methods (simple baselines, B6, B9, Ours); results C1–C3; pi0.5 as C4; limitations. | Full draft. |
| Dec 5–6 | Optional: the closed-loop check (run the top 2 methods live on fresh shelf starts, confirm they match the offline scores). Otherwise, revise. | |
| Dec 7–8 | Buffer, final edits, submit. | Submitted. |

## 5. Stop doing (cut)

- More pi0.5 training or data collection. Colab units are limited and it will not change C1–C3.
- A CNN/RNN help model trained with RL. B9 covers the RL comparison.
- The INSPECT action, new tasks or objects, and real-robot deployment.
- Rewriting the old plans (`PLAN.md`, `EXPERIMENT_PLAN.md`). This file supersedes them for the final week.

## 6. Limitations to write up (not to fix)

- The main result rests on a scripted expert, not a learned policy.
- Difficulty labels are defined by outcome (did any continuation succeed within 3 attempts?).
- Simulation only. The kitchen object set has no cups, so held-out generalization is the orange and the metal spoon.
- Myopic expected utility (B7) never asks at the default costs, by construction (it looks only one attempt ahead), so it
  matches never ask.
- On the drawer every method is within 0.4 reward of the clairvoyant, so it cannot separate the methods.
