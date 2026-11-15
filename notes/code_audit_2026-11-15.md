# Code audit: do our experiments intervene where we think they do? (2026-11-15)

Scope: the inference path and the training loss of each model, read in the
forks we run (`Coldilger/F1-VLA @ bridge-finetune`, `Coldilger/mimic-video @
main`, `Coldilger/LDA-1B @ bridge-finetune`). Not a line-by-line review of
every file; what was not read is listed at the end.

## F1-VLA

Read in full: `f1_vla/src/models/modeling_f1.py`; parts of `f1_policy.py`,
`wm/quant.py`, `wm/vqvae.py`.

| What we assumed | What the code does | Status |
|---|---|---|
| The forecast is decoded to an image | `fhat_to_img` is commented out (L636); `pred_imgs=None` | checked in code |
| The forecast is 10 scales | `num_resolutions = 4`: the loop stops after scale 4 (L610–611, same in training L311–315) | checked in code |
| The action expert reads 30 forecast tokens | It reads the keys/values of **29 forecast positions**. They hold the *inputs* of scales 2–4, which are built from the tokens sampled for scales 1–3 (1 + 4 + 9 = 14 tokens). The 16 tokens sampled for scale 4 are never fed back. | code + data: between two sampling seeds only the last 25 of the 709 gen positions differ (probe 767097, 15 decisions) |
| gen block = 30 history tokens + 679 forecast tokens (Experiment 1 internals) | gen block = **sos + 679 history-frame tokens, then 29 forecast positions** (`embed_wm_inputs` L268–276) | **our internals split was wrong** — see below |
| The action expert sees the forecast only through the KV cache | Yes (`denoise_step` reads `past_key_values`); the robot state enters separately through `state_proj` | checked in code |
| Interventions match the training interface | In joint training the action expert also reads its own *sampled* forecast (no teacher forcing in that branch, L383–470), so replacing the gen block at inference stays on the trained interface | checked in code |

Consequences for existing results (to be corrected in the F1 README and
thesis §5.1 once the old records are re-split):

- "28% of the action queries' attention goes to the generated future
  tokens" — that share was measured on a slice that is mostly history
  tokens.
- "The wrong-episode forecast changes the future tokens' keys/values by
  1–3%" — diluted by ~650 unchanged history positions; the change on the
  real forecast positions is larger.
- "The task-average forecast changes them by up to 38%" — the mean replaced
  the whole gen block, history included.
- Unaffected: the closed-loop results (139/288 with the task-average block,
  100/288 with the block removed) and the conclusion that the action does
  not depend on the episode-specific content of the gen block. It now reads:
  neither the history-frame tokens nor the forecast positions matter.

## mimic-video

Read: `cosmos_predict2/models/world2action_model.py` (training step,
`get_crossattn_emb`), `pipelines/video2world2action.py`. `video2world.py`
was read in earlier sessions, not again today.

| What we assumed | What the code does | Status |
|---|---|---|
| The decoder is conditioned on video-model activations of one layer after k denoising steps | `generate_video(..., return_context_at_step, hidden_state_layer_idx)` → `world2action_pipeline(crossattn_emb, context_timesteps)` | checked in code |
| In training the context comes from the real future video noised to a random sigma | `get_crossattn_emb`: real video + noise at `draw_video_sigma`, one forward pass of the video model | checked in code |
| — | The decoder also receives the sigma value itself, so it knows how noisy its context is | checked in code; not used in any experiment yet |

No discrepancy found in what was read.

## LDA-1B

Read: `lda/model/framework/QwenMMDiT.py` in full; the task routing, loss
and `predict_action` of `lda/model/modules/action_model/MMDiT_ActionHeader.py`;
the RoboCasa checkpoint's `config.yaml`.

| What we assumed | What the code does | Status |
|---|---|---|
| The forecast is a "dormant path" of the model | The head is trained on four tasks with equal weights: `policy`, `forward_dynamics`, `inverse_dynamics`, `video_gen`. The authors' inference uses `policy` only. Our "forecast on" condition is **our composition**: `video_gen()` output fed into the `inverse_dynamics` task. Both parts are trained; their chain is not, and in training `inverse_dynamics` only ever saw the real next observation. | checked in code — wording in thesis §3.3/§5.1 to be made explicit |
| The inverse-dynamics mode is trained for the whole action chunk | `future_obs_index = 16 = action_horizon`, so yes | checked in config |

## Not read / not checked

- F1-VLA: `paligemma_with_expert.py` (attention implementation), data
  pipeline, training script.
- mimic-video: `video2world.py` (today), the action decoder network itself,
  data pipeline.
- LDA-1B: the MM-DiT blocks, the data loader, the pretraining data mix.
- None of the three was compared with its paper text line by line.
