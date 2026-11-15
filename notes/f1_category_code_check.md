# F1-VLA: does the code match "Render-and-Decode"? (checked 2026-11-15)

Checked against `Coldilger/F1-VLA`, branch `bridge-finetune`, and our
checkpoint's `~/f1_vla_models/bridge_finetune/config.json`. Read from code,
not yet confirmed by a run.

| Claim in the draft | Code | Verdict |
|---|---|---|
| The pixel decoder is not used at inference | `modeling_f1.py` L636: `# pred_imgs = self.vae.fhat_to_img(f_hat)` is commented out; `pred_imgs=None` is returned (L640). `fhat_to_img` (`vqvae.py` L62) is the only path to pixels and has no other caller in `f1_vla/src`. | correct |
| The action expert sees the foresight only through attention / KV cache | Generated scales are appended to `past_key_values` (L575–583, `cat_past_key_values=True`); the action denoising loop reads only `past_key_values` (L621–627, `denoise_step`). | correct |
| Scale ladder `1_2_3_4_5_6_8_10_13_16`, 10 scales | `config.json`: `gen_expert_config.pn = "1_2_3_4_5_6_8_10_13_16"`; `vqvae.py` L25 default is the same. | correct for the ladder, **but see below** |
| One learned codebook for all scales | `quant.py` L38: a single `nn.Embedding(4096, 32)`; it comes from the pretrained VAR VQ-VAE (`vae_ch160v4096z32.pth`) and is frozen in F1 (`f1_policy.py` L63–64). The residual convolutions φ are partially shared: 4 for 10 scales (`share_quant_resi = 4`). | correct; "learned" = pretrained and frozen |

## What the draft misses: only 4 of the 10 scales are generated

`gen_expert_config.num_resolutions = 4` — in our checkpoint and in the
authors' own `f1_vla/config/f1_config.json` (L71). The generation loop
breaks after scale index 3 (`modeling_f1.py` L610–611), and training uses the
same cut (L311–315).

So the foresight the action expert reads is scales 1, 2, 3, 4:
1 + 4 + 9 + 16 = **30 tokens**, the finest being a 4×4 grid — not the full
16×16 map (680 tokens over 10 scales) that the VQ decoder would need to
render an image.

Consequence for the taxonomy: in the released code F1 neither decodes
pixels nor produces a latent that *could* be decoded into a full image. In
the survey's own terms it behaves like a Latent-Only model with a very
coarse future.

Not checked: the pixel size of the world-model images (the draft's
"256×256").
