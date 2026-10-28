# Does Foresight Drive Action? Testing Whether Inference-Time World Models Are Causally Used by Vision-Language-Action Policies (working title)

*Working draft — skeleton. Each section holds a short placeholder text that
says what the section will argue. Numbers are copied from the experiment
records in the three model repositories (linked per section); where a
result is under re-check, the text says so.*

---

## Abstract

Several recent robot policies generate a prediction of the future — an
image, a video, or a latent forecast — before choosing an action. They
report better performance than action-only policies, but it is not clear
*why*. Either the policy actually reads its own forecast when it decides
what to do, or training to forecast simply teaches better internal
representations, and the forecast itself is unused at inference time.

This thesis separates these two explanations with one shared set of five
experiments, applied to three architecturally different models: F1-VLA,
mimic-video, and LDA-1B. We remove or replace the forecast (ablation),
feed the true future instead (oracle injection), measure what the forecast
costs (latency), probe what the internal representation encodes
(probing), and erase confounding information from it (concept erasure).

*Headline finding (to be finalized):* the three models disagree. F1-VLA
needs *some* real visual signal in its forecast slot but barely cares about
its content; mimic-video collapses whenever the forecast is wrong or
missing (under re-check, see Chapter 5); LDA-1B does not use its forecast
at inference at all, and turning it on does not help.

---

## 1. Introduction

### 1.1 Motivation
Robot foundation models increasingly "imagine before they act". This is
expensive: generating a forecast can multiply the time per decision. If
the forecast is not actually used, that compute is wasted, and the claimed
benefit of world models is really a training-time effect.

### 1.2 Research question
**Does the world-model computation performed at inference time carry
causal weight for action selection — or is the benefit entirely a
training-time representational effect?**

### 1.3 Approach in one paragraph
Instead of comparing models by their final success rate, we intervene on
the forecast inside each model and watch how the actions change, in
closed-loop simulation. Causal claims come only from closed-loop success
(or from actions compared at identical states), never from offline
prediction error on training data, which memorization can explain.

### 1.4 Contributions
1. A shared, model-agnostic protocol of five experiments for testing
   whether a policy uses its own foresight.
2. Its application to three models with different foresight mechanisms
   (image tokens, video-diffusion features, latent dynamics).
3. Evidence that "uses its world model" is not one property: the three
   models depend on foresight in qualitatively different ways.
4. Negative and methodological results recorded as findings, e.g. that
   LDA-1B's representation does not transfer across camera viewpoint.

### 1.5 Thesis outline
Chapter 2 reviews related work, Chapter 3 the three models, Chapter 4 the
shared protocol, Chapter 5 the results, Chapter 6 the follow-up
experiments, Chapter 7 discussion and limitations.

---

## 2. Background and Related Work

### 2.1 Vision-language-action policies
Policies that map camera images and a language instruction to robot
actions, usually built on a pretrained vision-language model. Key
reference point: π0 (Black et al., 2024), on which F1-VLA builds.

### 2.2 World models and foresight inside policies
Models that predict future observations and condition actions on them:
F1-VLA (arXiv:2509.06951), mimic-video (arXiv:2512.15692), LDA-1B
(arXiv:2602.12215). mimic-video's own paper already runs a version of our
oracle experiment (Section III, Fig. 2) — our Experiment 2 extends it.

### 2.3 Causal interventions on neural networks
Ablation and activation patching (Heimersheim & Nanda, 2024), interchange
interventions (Geiger et al., 2021), and the warning that zero-ablation can
push a network off its activation distribution.

### 2.4 Probing and concept erasure
Linear and nonlinear probes for what a representation encodes; LEACE
concept erasure (Belrose et al., 2023); amnesic probing (Elazar et al.,
2021) as the causal follow-up.

### 2.5 Evaluating robot policies in simulation
SimplerEnv (Li et al., 2024) for BridgeData V2 tasks (Walke et al., 2023);
RoboCasa (Nasiriany et al., 2024) for LDA-1B. Statistical reporting
following Agarwal et al. (2021).

*TODO: verify every citation against the published version and build the
bibliography.*

---

## 3. Models Under Study

One short section per model: what the forecast is, where it enters the
action decoder, and what "removing it" means mechanically.

### 3.1 F1-VLA
Generates a foresight image with a next-scale (VAR) generator at every
control step; the action expert reads it through a shared KV cache.
Finetuned by us on Bridge (chunk size 4, 100k steps).

### 3.2 mimic-video
A video-diffusion backbone (Cosmos-Predict2) is partially denoised; hidden
activations from one layer (`crossattn_emb`) condition a flow-matching
action decoder through cross-attention. Operating point: one denoising
step (`stop=1`).

### 3.3 LDA-1B
A shared MM-DiT trained with a visual-forecasting co-objective, but the
default inference path does not run the forecast. Bridge finetunes reached
0%; experiments run on the authors' RoboCasa checkpoint (48% baseline).

---

## 4. Shared Experimental Protocol

### 4.1 Benchmarks and tasks
SimplerEnv-Bridge, four tasks × 24 episodes (F1-VLA, mimic-video);
RoboCasa-GR1 tabletop tasks, 50 episodes (LDA-1B).

### 4.2 The five experiments
1. **Ablation** — remove or replace the forecast; measure closed-loop
   success.
2. **Oracle injection** — feed the true future through the model's own
   pipeline; compare the resulting actions.
3. **Cost per decision** — latency with and without the forecast.
4. **Representation probing** — can current and future end-effector pose
   be decoded from a frozen hidden state?
5. **Concept erasure** — erase episode identity (LEACE); does pose become
   decodable?

### 4.3 What counts as evidence
Closed-loop success, or actions compared at identical states. Offline
prediction error on training episodes is not used anywhere (no held-out
split, memorization-confounded).

### 4.4 Statistics (shared rule for all experiments)
Every condition runs on the same task instances and simulator seeds as
its baseline. We report successes out of total trials, paired differences
between conditions, and confidence intervals (Agarwal et al., 2021). Small
gaps such as 48.6% vs 50.0% are only interpreted if the interval excludes
zero. *Older runs predate this rule; their baselines will be re-run.*

### 4.5 Reporting by task type (shared rule)
Results are reported per task and per task type, not only as an average.
SimplerEnv-Bridge offers only four short pick-and-place tasks (stacking is
the most contact-heavy), so a full split into reaching, contact-heavy and
long-horizon tasks is only possible on RoboCasa.

---

## 5. Results

### 5.1 Experiment 1 — Ablation
- **F1-VLA:** zeroing the foresight lowers success from 48.6% to 34.7%
  (−13.9 pp); a real image from a wrong episode recovers 50.0%. The model
  needs *some* real visual signal, not the right one.
- **mimic-video:** every intervention on the forecast gives 0% (baseline
  46/96 = 47.9%, rerun on identical episodes). A sanity check requested by
  the advisor shows that plain zeroing is not clean evidence: it makes all
  context tokens identical, cross-attention uniform, and the decoder's
  rotations invalid — the computation breaks. But controls that keep the
  decoder on its normal activation range collapse just as completely:
  the task-average forecast (0/96; the robot barely moves), a
  coherent clip from another episode, and the original per-step shuffle
  (both 0/96, paired difference −47.9 pp, 95% CI [−56.2, −39.6]). Under
  the shuffles the robot moves normally but never touches the correct
  object. Reusing the episode's own first forecast for the whole episode
  also fails (0/96), though it is the only intervention that still
  reaches the correct object sometimes (12/96). Conclusion: the decoder
  follows the *content* of a correct and current forecast.
- **LDA-1B:** turning the dormant forecast *on* gives 44% vs 48% baseline,
  within noise.

### 5.2 Experiment 2 — Oracle injection
- **F1-VLA:** the oracle's action is close to the real policy's own
  action — consistent with content mattering little.
- **mimic-video:** the oracle's action is ~5–9× closer to the policy's own
  action than a zero-action guess; with Experiment 1, this suggests the
  model's own forecast is already close to correct.
- **LDA-1B:** oracle and forecast actions are both worse than a zero-action
  baseline (0.90 vs 0.75 L1), and far from each other: the content of the
  fed-in future changes this pathway's output.

### 5.3 Experiment 3 — Cost per decision
F1-VLA 215.7 ms median per control step; mimic-video 1060 ms per decision
at `stop=1` (~4–5× F1/LDA); LDA-1B 254.7 ms, rising to 428.2 ms (+68%)
with the forecast turned on.

### 5.4 Experiment 4 — Representation probing
F1-VLA and mimic-video: current pose is not recoverable (worse than a
constant predictor), with linear and MLP probes alike. LDA-1B: an MLP
recovers both current and future pose where ridge regression fails.

### 5.5 Experiment 5 — Concept erasure
F1-VLA and mimic-video: erasing episode identity does not make pose
recoverable — the "masking" explanation is refuted. LDA-1B: most of the
MLP's current-pose signal turns out to be a linear direction ridge missed;
future pose keeps a smaller nonlinear residual.

### 5.6 LDA-1B and camera viewpoint
Swapping RoboCasa's egocentric camera for a third-person one drops success
from 48% to 0/50, with a self-occlusion caveat. Together with the Bridge 0%,
this supports "LDA-1B does not transfer across viewpoint".

*Sources: `eval/bridge/EXPERIMENTS.md` and `experiment*/README.md` in each
repository; `LDA-1B/eval/robocasa/RESULTS.md`.*

---

## 6. New Experiments (from the advisor's list, in progress)

Follow-ups that ask a new question get their own experiment number. The
ablation sanity check stays part of Experiment 1, and paired statistics
and per-task reporting are shared rules (Sections 4.4–4.5).

### 6.1 Experiment 6 — Graded foresight replacement
Replace the native forecast with one from the same task and scene, a
different task in the same scene, or an unrelated scene. Compare actions
at identical saved states, then run the most informative conditions in
closed loop. Extends Experiment 1's shuffle into a scale of "how wrong";
method: interchange interventions (Geiger et al., 2021).

### 6.2 Experiment 7 — Horizon-shifted foresight
Inject futures predicted for earlier or later horizons than the action
branch expects: does the *timing* of the predicted event matter?

### 6.3 Experiment 8 — Instruction–foresight conflict
Keep the observation, change the instruction, compare futures and actions;
then swap the futures between instructions. Does the action follow the
goal, the foresight, or both? (Cue-conflict design, Geirhos et al., 2019.)

### 6.4 Experiment 9 — Compute–performance trade-off
Vary generation steps, prediction detail and replanning frequency; plot
task success against full inference latency. Builds on Experiment 3 and on
mimic-video's own denoising-depth analysis.

### 6.5 Experiment 10 — Perturbed starts and object displacement
Evaluate from perturbed starting states or after moving an object mid-
episode: is foresight most useful when the robot has to adapt?

### 6.6 Experiment 11 — Matched action-only baseline (if compute permits)
Same data and training budget without a generation branch (for F1-VLA:
π0), compared with the jointly trained model with and without its
generation branch at inference.

### 6.7 Experiment 12 — BridgeData pipeline audit
Episode overlap, language annotations, frame sampling, action coordinates
and orientation corrections across the two Bridge processing pipelines
(`bridge_orig_lerobot` and LDA-1B's converted copy).

---

## 7. Discussion

### 7.1 Three models, three kinds of dependence
Structural (F1-VLA), content-sensitive (mimic-video, pending re-check),
and none at inference (LDA-1B).

### 7.2 What this means for "world models help robots"
The benefit can come from training, from inference, or both — and which
one is model-specific.

### 7.3 Limitations
Simulation only; one benchmark per model; small episode counts; LDA-1B
evaluated on RoboCasa rather than Bridge; our F1-VLA finetune is
undertrained relative to the paper (0.85 vs ~10 epochs).

---

## 8. Conclusion

*One page: the answer to the research question for each model, and what a
future world-model policy should be tested for.*

---

## Appendix

- **A. Reproducibility.** Repositories, branches, checkpoints, and the
  scripts that rebuild every environment and dataset.
- **B. Debugging records.** Operating-point correction for mimic-video
  (`stop=1` vs `stop=23`), LDA-1B's Bridge 0% diagnostic trail.
- **C. Negative results.** chunk size 8/30 for F1-VLA; offline probes
  (excluded from evidence, kept for the record).
