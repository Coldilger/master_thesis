# The three papers against the code and our experiments (2026-11-15)

Read from the arXiv HTML versions, by section (method, training/inference,
ablations, limitations), not cover to cover. Line references are to the
code in our forks.

## F1-VLA — Lv et al., 2025, "F1: A Vision-Language-Action Model Bridging Understanding and Generation to Actions", arXiv:2509.06951

| Paper | Code / our results |
|---|---|
| Foresight tokens "are subsequently decoded into the predicted future image"; the action expert is conditioned on "the generated foresight image" | The decoder call is commented out; the action expert reads keys/values of the forecast positions (`modeling_f1.py` L636, L621–627). **Differs.** |
| Stage I trains the generator "across 10 resolutions"; "for efficient real-time control during inference, foresight prediction is restricted to 4 scales" | `num_resolutions = 4`. **Agrees** — the 4-scale cut is the authors' documented choice, not a quirk of the release. |
| Attention: generation attends to understanding, action attends to both, nothing flows back | Matches the cache structure (und → gen → action). |
| Ablation Table 4 (LIBERO + SimplerEnv-Bridge): full model vs. *2-Scales* and *6-Scales* **at inference time**, and *No-Gen* (generation expert removed, trained without it) | Bridge columns are multiples of 1/24, i.e. 24 episodes per task, one run: full 70.8 / 66.7 / 50.0 / 50.0 (mean 59.4), 2-Scales mean 52.1, 6-Scales mean 57.3, No-Gen mean 27.1. Differences of 2–7 pp on 96 episodes are inside the noise we measure (Wilson ±10 pp). |

What this gives us:

- **Advisor's point 4 for F1 has a direct precedent**: the authors vary the
  number of scales at inference (2 / 4 / 6). We can repeat it with our
  paired protocol (288 episodes) and add latency.
- **Advisor's point 5 has a published data point**: No-Gen. It is confounded
  (that variant also skips the pretraining stages), so it does not replace
  a matched baseline, but it must be cited.
- Our finetune reaches 48.6% on Bridge against the paper's 59.4%.

## mimic-video — Pai et al., 2025, "mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs", arXiv:2512.15692

| Paper | Code / our results |
|---|---|
| The action decoder is described as an inverse dynamics model on intermediate video-model representations; 5 context frames; partial denoising to an intermediate flow time | Matches (`video2world2action.py`, `world2action_model.py`). |
| Decoder is trained on representations of the **ground-truth** future video, with independent flow times for video and action | Matches `get_crossattn_emb`. The decoder also receives the video flow time through AdaLN. |
| SIMPLER-Bridge, Table I: 37.5 / 37.5 / 12.5 / 100.0, mean 46.9% (Carrot / Spoon / Stack / Eggplant) | Ours: 37.5 / 50.0 / 16.7 / 87.5, mean 47.9%. **Reproduced.** |
| Fig. 6: success vs. video flow time on SIMPLER-Bridge peaks at the noisiest setting; fully denoising hurts | Our Experiment 9 closed loop reproduces the shape with paired statistics. |
| Their two explanations (App. E): (1) a fully denoised video can be a wrong plan or subtly out of distribution for a decoder trained on ground truth — "noise as augmentation"; (2) near the clean end the hidden states approach an identity map and carry less | Our seed-averaging test separates part of (1): averaging the action over four imagined futures at k=16 restores success, so the loss comes from committing to one imagined future, not from the decoder being unable to use deep contexts. Explanation (2) is not tested by us. |
| Sec. III case study: with ground-truth ("oracle") future latents success is near perfect | Our Experiment 2 for mimic-video is a weaker, single-step version of this. |

## LDA-1B — Lyu et al., 2026, "LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion", arXiv:2602.12215

| Paper | Code / our results |
|---|---|
| One MM-DiT co-trained on policy, forward dynamics, inverse dynamics and visual forecasting; a task embedding selects the objective; "at inference time, the same model can be flexibly invoked for different objectives" | Matches `MMDiT_ActionHeader.py`. Our "forecast on" condition (visual forecasting → inverse dynamics) uses this mechanism, but the chain itself is ours and is not evaluated in the paper. |
| In policy mode the future-state slot holds a register token | Matches (learnable placeholder tokens). |
| Forecasting in DINO latent space; VLM and DINO frozen | Matches. |
| Pretraining data EI-30k (Table IX): Open X-Embodiment 3000 h, Agibot World 3276 h, other real robots ~2300 h, simulation 8600 h, egocentric human video ~16,600 h | **Third-person robot data is in the pretraining mix** (Open X-Embodiment, which contains BridgeData V2), about a tenth of the total. |
| Limitations: "predominantly egocentric camera viewpoints, which may constrain generalization to new visual perspectives" | The authors state the viewpoint limitation themselves. Bridge is not evaluated. |

Consequence for the Bridge question: the model has seen third-person robot
frames in pretraining, so "it never saw this viewpoint" is not the
explanation for our 0%; a data-pipeline difference (advisor's point 9)
becomes the first thing to check. See `lda_bridge_viewpoint.md`.
