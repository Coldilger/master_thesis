# LDA-1B on Bridge: what is known, what is not (2026-11-15)

## Facts

- F1-VLA and mimic-video were published with Bridge results; LDA-1B was
  not evaluated on Bridge (arXiv:2602.12215), although Open X-Embodiment is
  part of its pretraining data.
- Our Bridge finetunes of LDA-1B gave 0% on SimplerEnv Bridge (summer 2026;
  diagnostic trail in `LDA-1B/eval/bridge/RESULTS.md`).
- The authors' RoboCasa checkpoint works in our setup (48% on
  `PnPCupToDrawerClose`), so the codebase and evaluation are not broken.
- RoboCasa-GR1 feeds an egocentric head camera; the paper describes its
  benchmarks and real-robot runs as egocentric.
- Camera test (2026-08-18, `LDA-1B/eval/robocasa/RESULTS.md`): same
  checkpoint, task and episodes, camera switched from `egoview` to
  `robot0_agentview_center` → 0/50. Caveat recorded then: this camera sits
  behind the robot, which blocks part of the scene.

## What the camera test does and does not show

It shows that a checkpoint finetuned only on egocentric frames does not
work zero-shot from another viewpoint. Most policies would fail this test.
It does **not** show that LDA-1B cannot *learn* a third-person view: in the
Bridge finetune the model was trained on third-person frames, and still
gave 0%. So the camera test alone does not explain the Bridge result.

Two explanations remain for Bridge 0%:

1. **Viewpoint / pretraining.** The pretrained model has seen (almost) only
   egocentric data and our finetune budget was too small to move it.
2. **Data pipeline.** Something differs between our converted Bridge copy
   and what the model expects (actions, frames, language) — the advisor's
   point 9 (Experiment 12).

## Cheap checks, in order

1. ~~List the pretraining datasets and their camera types.~~ Done from the
   paper (2026-11-15): EI-30k includes Open X-Embodiment (3000 h, contains
   BridgeData V2) and other third-person robot data, about a tenth of the
   total; the paper's own limitations name "predominantly egocentric camera
   viewpoints". So third-person frames were seen in pretraining, which makes
   explanation 2 the first to check.
2. Experiment 12: audit the two Bridge pipelines (`bridge_orig_lerobot` vs.
   `convert_bridge_to_lda.py`).
3. RoboCasa camera sweep: two or three third-person cameras that do not
   occlude the scene, inference only. Shows whether 0/50 was the viewpoint
   or the occlusion — but still only the zero-shot question.

Expensive, only if 1–2 point to viewpoint: finetune on third-person data
inside RoboCasa (needs re-rendered demonstrations), or a new Bridge
finetune.

## Consequence for the thesis

LDA-1B is evaluated on RoboCasa, the other two on Bridge (limitation,
§7.3). Whatever the cause of Bridge 0%, it has to be described as "not
resolved", not as "LDA-1B does not transfer across viewpoints".
