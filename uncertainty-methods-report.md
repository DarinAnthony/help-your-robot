# Estimating uncertainty in task completion: methods for deciding when a robot should ask for help

*Deep-research report, 2026-11-13. The workflow searched along 5 angles, fetched 22 sources, pulled 104 claims from them, and ran the top 25 claims through 3-vote adversarial checking: 24 were confirmed and 1 was refuted. Claims marked **[source-read]** were taken from the primary source but did not go through the 3-vote check.*

## Bottom line

- **Semantic/intent uncertainty is mostly solved.** Conformal prediction over an LLM's multiple-choice plan options (KnowNo) and its refinements, Introspective Planning and CoFineLLM, give calibrated "ask vs. act" decisions with a coverage guarantee.
- **Capability and outcome uncertainty is where the open problem is.** KnowNo explicitly assumes that actions execute successfully. The strongest 2025–26 signals are calibrated probes on VLA internals (VLAConf, SAFE), failure detectors trained only on successful runs (FAIL-Detect, FIPER, Sentinel), and VLM failure detectors (AHA, Guardian). Each has gaps.
- **VLM success detectors are weak on contact-rich tasks.** On FailBench they score near chance on these, which is a warning for jar opening.
- **Raw token entropy is a poor failure signal.** Mean entropy over a rollout can be below chance on long-horizon tasks, and raw diffusion-policy variance does no better.
- **Almost every recent method decides with a threshold or budget rule on calibrated confidence.** The sequential and cost-sensitive work is either old POMDP work (Rosenthal & Veloso; Chung et al.) or a small recent set: offline RL in Jiang et al., cost models in Banerjee et al., and RL fine-tuning in Ask-to-Act. A belief-based POMDP that combines calibrated semantic and capability signals, compared against those threshold methods, is a real gap.

---

## 1. Methods ranked by uncertainty type

### A. Semantic / intent uncertainty (which plan or object did the user mean?)

| Rank | Method | Calibrated? | Low-data? | Class-project fit |
|---|---|---|---|---|
| 1 | **KnowNo**: conformal prediction over LLM multiple-choice options (Ren et al., CoRL 2023 Oral) | Yes, marginal coverage under exchangeability | Needs about 400 labeled calibration examples (about 4 hours of labeling) | **High**. Simple, and works with any LLM that exposes logprobs |
| 2 | **Introspective Planning**: retrieval of LLM-generated rationales plus conformal prediction (Liang, Zhang, Fisac, NeurIPS 2024) | Yes, with tighter sets than KnowNo | Needs about 200–400 human-labeled tasks | Medium |
| 3 | **LAP**: LLM probability × affordance "A-Feasibility" score, then KnowNo-style conformal prediction (Mullen & Manocha, arXiv 2403.13198) | Yes, via conformal prediction | Same as KnowNo | **High**. The prompt-based feasibility term is cheap |
| 4 | **CoFineLLM**: fine-tuning that is aware of conformal prediction, with loss L_CE + λ·L_CP (Wang, Vorobeychik, Kantaros, arXiv 2511.06575) | Yes | Needs fine-tuning (4k training scenarios) | Low (cite as related work) |
| 5 | **Ask-to-Act / AutoAsk**: multimodal LLM fine-tuned with online RL to ask clarifying questions (Ramrakhya et al., arXiv 2504.00907) | No; this is a learned policy | Expensive | Low (cite as the learned-policy precedent) |
| — | **Semantic entropy** (Kuhn, Gal, Farquhar, ICLR 2023) | No guarantee | Needs no labels | Medium. *Not checked in this run; the citation is well known.* |

**Verified numbers**

- **Introspective Planning** (Safe Mobile Manipulation, GPT-4 Turbo, 85% target): help rate 63.0 vs. 77.5 for KnowNo, over-ask rate 6.4 vs. 51.3, success 87.5 vs. 84.5.
  - *Refuted (0–3):* the claim that it "beats KnowNo on every metric." Describe success as comparable, not better.
- **LAP** (real mobile manipulator, 70% success): help rate fell from 63% (KnowNo) to 42%, a 33.3% reduction. AuC was 0.72 vs. 0.67. Your preliminary claim of ">33%" holds, but only just. The numbers are read off a figure at a single operating point, and the comparison is essentially against KnowNo alone.
- **Ask-to-Act**: success on unseen scenes / unseen tasks was 45.9 / 38.7, vs. 33.6 / 29.5 for LLaVA-OV SFT and 15.7 / 14.3 for GPT-4o+SoM+ReAct. This claim passed 2–1, not unanimously. Baselines with privileged full observability score higher (46.7 / 44.4).
  - **Citation fix:** the paper's title is *"Grounding Multimodal LLMs to Embodied Agents that Ask for Help with Reinforcement Learning"*. "Ask-to-Act" is the name of the task, and no venue is listed.
- **CoFineLLM** (BabyAI-Text, 0.95 target coverage) **[source-read]**: help rate fell from 12.7% to 9.7% at equal coverage. In an out-of-distribution test it fell from 11.7% to 6%.

**Key limitation (verified, 3–0).** KnowNo assumes three things: the scene is fully grounded in the LLM's text, the chosen actions execute successfully, and humans answer faithfully. It cannot express "I understand the request but can't do it." Jar opening needs a separate capability belief.

### B. Capability / feasibility uncertainty (can I physically do this?)

| Rank | Method | Calibrated? | Low-data? | Class-project fit |
|---|---|---|---|---|
| 1 | **Learned success probe + Platt scaling** (VLAConf, Huang et al., arXiv 2605.29605). Stage 1 trains a Coin-Flip Network on frozen VLA features using successful runs only. Stage 2 fits 2-parameter Platt scaling on a few labeled rollouts | Yes. A plain logistic-regression probe is about as well calibrated (NLL 0.253 vs. 0.255) | **Yes** | **High**. The recipe carries over to any feature vector |
| 2 | **LAP prompt-based feasibility**: P("possible and safe") from the LLM, optionally × binary context feasibility | Only after calibration | Yes | **High**. Cheapest prior available |
| 3 | **SayCan**: LLM usefulness × learned skill value function (Ahn et al., 2022) | No | Needs learned value functions | Medium (conceptual anchor). *Not checked in this run.* |
| 4 | **Prompt-ensemble VLA calibration**: average action-token confidence over 20 rephrasings of the instruction (Zollo & Zemel, arXiv 2507.17383) | Improves ECE by up to about 40%. ECE2 and NLL improved in every setting; ECE1 and Brier in 5 of 6 | Training-free | Medium. Works only for VLAs that expose token probabilities |

**VLAConf numbers** (π0.5 on LIBERO, online): Brier 0.068, NLL 0.255, failure AUROC 0.876. For comparison, logistic regression reached 0.731 AUROC and DeepSVDD 0.571. So VLAConf's advantage is in *discrimination*; calibration comes mostly from the Platt step. With OpenVLA-OFT, ConfidenceVLA has lower online ECE (0.028 vs. 0.119).

### C. Outcome / execution uncertainty (is this attempt failing?)

| Rank | Method | Output | Notes |
|---|---|---|---|
| 1 | **FAIL-Detect** (Xu et al., TRI, RSS 2025). Sequential OOD detection on policy inputs and outputs, trained only on successful runs, with time-varying conformal thresholds | Calibrated *false-alarm rate*, not P(success) | Peer-reviewed. A flow-based density score was the most consistent |
| 2 | **SAFE** (arXiv 2506.09937) **[source-read]**. Probe on VLA features plus functional conformal prediction | Score + conformal band | About 81–82 ROC-AUC on seen tasks and 77–78 on unseen tasks across OpenVLA, π0-FAST, π0 and SimplerEnv. On a real Franka it drops from 86.8 to 64 on unseen tasks. Adds under 1% inference overhead |
| 3 | **FIPER** (Römer et al., NeurIPS 2025) **[source-read]**. Random network distillation for OOD observations, plus action-chunk entropy | Score | Needs no failure data |
| 4 | **Sentinel** (Agia et al., CoRL 2024) **[source-read]**. STAC temporal-consistency score for erratic failures, plus a VLM monitor for task-progress failures | Detection, then stop | Shows that raw diffusion-policy variance is a poor detector |
| 5 | **AHA** (Duan et al., ICLR 2025). Open-source VLM failure reasoner | Binary + explanation | Raised downstream task success by 21.4% over GPT-4 feedback. Not calibrated |
| 6 | **Guardian** (Pacaud et al., arXiv 2512.01946). InternVL3-8B with LoRA, checks plans and each subtask | Binary | Modest gain in simulation (0.45 to 0.54). The dataset is GuardianFail-36k; "FailCoT" is the wrong name |

**Two important negative results**

- **FailBench** (arXiv 2609.03611, Sept 2026) tested 13 VLM success detectors on 2,197 attempts.
  - The best detector (Gemini 3 Flash) reached only **0.77 balanced accuracy**.
  - Detectors are **near chance (≤0.60) on contact-heavy tasks**, and lean toward predicting success when evidence is ambiguous.
  - Specialist failure-detection fine-tunes, including Guardian, scored 0.50–0.63, *below* general-purpose VLMs.
  - **Implication:** shelf retrieval is fine, because a successful grasp visibly moves the object. A failed jar twist looks like nothing happened (this is an extrapolation).
- **Token entropy** (arXiv 2603.18342) **[source-read]**. Mean token entropy over an OpenVLA rollout gave AUROC **0.468 on LIBERO-10** (below chance) and 0.845 on LIBERO-SPATIAL. Uncertainty is concentrated in time around the failure. Max-pooling over a sliding window plus calibration raised AUROC to 0.84–0.94.
  - This backs up the concern in your preliminary notes about the Karli et al. π0-FAST entropy paper. That paper could not be fetched in this run.

---

## 2. Sequential and cost-sensitive decisions about when to ask

All of these are **[source-read]** except the VLAConf result.

- **HOP-POMDP** (Rosenthal & Veloso, CMU). A POMDP with an explicit *ask* action. A human answers with availability α_s and returns a null observation otherwise. Over 5,929 parameter settings, the availability-aware policy differed from the availability-blind one 39.7% of the time.
  - **Important for you:** **QMDP never chooses to ask**, because it assumes the state becomes fully observable after one step. You need a solver that values pure information-gathering, such as point-based value iteration (PBVI/SARSOP), POMCP, or explicit value-of-information lookahead.
- **Chung, Milliken, Hollinger & Tumer** (Oregon State). A POMDP with a hidden Novice/Expert *capability* state, updated from observations of team reward, with actions NoRequest and RequestAssistance. **This is the closest structural precedent to your capability belief.**
- **Jiang, Paris, Kulić & Tian** (CSIRO/Monash, 2026). Offline RL with Conservative Q-Learning (CQL) deciding whether to request a human waypoint, demonstrated on a Boston Dynamics Spot.
  - Area explored: 232.0 m² for offline RL, vs. 224.3 for LinUCB, 218.6 for the score-based policy, and 199.0 for the threshold rule.
  - Expert recovery time fell from 70.5 s to 29.1 s.
  - **Corrections to your preliminary claim:** the venue is *not confirmed* as RA-L; the project page says only "accepted paper, 2026". The "threshold" baseline is a heuristic stuck/inefficiency rule, not an uncertainty threshold. RL beat the contextual bandit only narrowly.
- **Banerjee, Palempalli, Yang, Fang, Abdullah, Silver, Dean & Bhattacharjee** (HRI '26, DOI 10.1145/3757279.3788668). Has a module selector plus a query policy that uses calibrated module-level uncertainty and a model of human cost. Applied to bite acquisition in robotic feeding.
  - **Your preliminary description holds.** Results: lower workload than Always Query, higher subjective success than Never Query, and better than ConfidenceQuery (two studies with N=10 each). Success rates appear only in figures.
- **VLAConf handoff** (verified). With help at about 10% of steps, the calibrated-confidence trigger reached 94.7% success. Comparisons: 88.7% with no help, 91.0% with a time-based trigger, 89.7% random, 92.3% always-help, and 97.7% for an expert-only upper bound. The curves are *non-monotonic*: at about 27% help, the time-based trigger wins. The thresholds were set after the fact, not online.

---

## 3. Status of your preliminary claims

| Claim | Status |
|---|---|
| KnowNo (Ren et al., CoRL 2023) | ✅ Verified (3–0). It was an Oral, and it extends conformal prediction to multi-step plans |
| LAP: >33% fewer help requests at 70% success | ✅ Verified, but just barely (63% to 42% = 33.3%). Venue not confirmed |
| Ask-to-Act (Ramrakhya et al.) | ✅ Verified. ⚠️ Fix the title (see §1A); there is no venue |
| Karli et al., "Ask Before You Act" (π0-FAST entropy) | ❓ Could not fetch the OpenReview PDF. Related work supports the direction of the finding |
| Banerjee et al., HRI 2026 | ✅ Read from the source (not 3-vote checked). Authors, venue and framing are correct |
| Jiang et al., RA-L 2026 | ⚠️ Venue not confirmed. "Uncertainty threshold" should read "heuristic rule threshold" |
| Semantic entropy (Kuhn, Gal, Farquhar, ICLR 2023) | ➖ Not checked in this run. A standard citation |
| SayCan | ➖ Not checked in this run. A standard citation (Ahn et al., 2022) |

---

## 4. Recommendations for the CS 238 design

**Belief components and their signals**

1. **Semantic belief over what the user meant.** Use KnowNo-style normalized multiple-choice probabilities from the LLM as the prior. Keep the conformal set as a *baseline*, not as the policy.
2. **Capability belief over P(feasible).** Use LAP-style P("possible and safe"), optionally multiplied by context feasibility. **Calibrate it** with Platt scaling or logistic regression on a few hundred labeled simulated rollouts, following VLAConf.
   - For the jar, include an explicit feature for torque or capability margin, so the prior can be confidently low. Low uncertainty that leads straight to asking is exactly the case your proposal highlights.
3. **Outcome observation model.** Treat the success detector as a **noisy sensor with an empirical confusion matrix**, P(detector says success | true outcome), not as ground truth.
   - Shelf: a VLM detector is fine.
   - Jar: use simulated proprioception or torque observations (for example, motor current saturated with no lid rotation). FailBench suggests VLM detection would be near chance here.
4. **Solver.** Use PBVI/SARSOP/POMCP, or a myopic value-of-information baseline. **Don't use QMDP for the main policy, because it never asks** (Rosenthal & Veloso). QMDP is still a useful *baseline* that shows this failure.

**Baselines**

- Never ask
- Always ask
- Random asking, at the same help rate as your policy
- KnowNo threshold: ask if the conformal set has more than one option
- LAP threshold
- Calibrated-confidence threshold or budget trigger, VLAConf-style
- Time- or retry-based trigger
- QMDP
- Myopic expected-utility rule using the same calibrated beliefs. This isolates the gain from sequential planning.
- Optional: LinUCB contextual bandit, following Jiang et al.

**Metrics**

- Success vs. help-rate curves, and pAUC over low help budgets, as in VLAConf and LAP
- Total reward under your cost model
- Brier, NLL, ECE and AUROC for each belief signal on its own, so you can separate predictor quality from policy quality

**Where the POMDP should win**

- (a) Shelf retrieval near the edge of the robot's reach, where a cheap grasp attempt has high value of information.
- (b) A jar that is clearly beyond the robot's torque, where it should ask before trying at all. A threshold on *uncertainty* gets this wrong; a threshold on *P(success)* gets it right. Make sure your baselines include both kinds.

## 5. Open questions

- Can a conformal coverage guarantee over semantic options be combined with a Bayesian capability belief without double-counting uncertainty or losing the guarantee?
- How can you calibrate capability uncertainty when a failure leaves no visual trace?
- Does a POMDP or VOI policy beat a calibrated-threshold policy at the *same* calibrated beliefs? No verified paper tests this directly, and it is your contribution.

## Caveats

- Many 2026 sources are arXiv preprints with self-reported results, mostly in simulation: VLAConf, FailBench, Guardian, Zollo & Zemel, CoFineLLM, SAFE and token-entropy.
- The peer-reviewed anchors are KnowNo, Introspective Planning, AHA, FAIL-Detect, Sentinel and FIPER.
- Conformal guarantees are marginal and assume exchangeability.
- Conclusions specific to the jar task are extrapolations.

## Sources

- KnowNo: https://arxiv.org/abs/2307.01928 · https://robot-help.github.io
- Introspective Planning: https://arxiv.org/abs/2402.06529
- CoFineLLM: https://arxiv.org/abs/2511.06575
- LAP: https://arxiv.org/html/2403.13198v2
- Ask-to-Act: https://arxiv.org/abs/2504.00907
- VLAConf: https://arxiv.org/abs/2605.29605
- Zollo & Zemel (VLA calibration): https://arxiv.org/abs/2507.17383
- FailBench: https://arxiv.org/abs/2609.03611
- AHA: https://aha-vlm.github.io/
- Guardian: https://arxiv.org/html/2512.01946
- FAIL-Detect: https://arxiv.org/abs/2503.08558
- Sentinel: https://arxiv.org/pdf/2410.04640
- SAFE: https://arxiv.org/html/2506.09937v1
- FIPER: https://arxiv.org/abs/2510.09459
- Token-entropy aggregation: https://arxiv.org/abs/2603.18342
- Banerjee et al. (HRI '26): https://arxiv.org/abs/2602.10289
- Jiang et al.: https://human-robot-teaming.github.io/
- HOP-POMDP (Rosenthal & Veloso): http://www.rosenthalphd.com/papers/Rosenthal_ROMAN11.pdf
- Chung et al.: https://jenjenchung.github.io/anthropomorphic/Papers/Chung2017ask.pdf
