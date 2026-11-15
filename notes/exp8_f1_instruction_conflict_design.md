# Experiment 8 (advisor's point 7) for F1-VLA — design under discussion (2026-11-15)

Not run yet. Extends the cue-conflict design of Geirhos et al. (2019) and
interchange interventions (Geiger et al., 2021) to a policy's forecast.

## Question

The instruction reaches F1-VLA's action expert by two routes: directly (the
language tokens sit in the understanding block of the shared KV cache) and
through the forecast (the generation block attends to the language tokens
while it is produced). When the two routes disagree, which one does the
action follow?

## 2 × 2 design on identical saved states

| | forecast generated under A | forecast generated under B |
|---|---|---|
| action expert reads instruction A | AA (normal) | AB (conflict) |
| action expert reads instruction B | BA (conflict) | BB (normal) |

Implementation: run the model twice on the same observation (instructions A
and B, same sampling seed, same action noise), then let the action expert
read the understanding block of one run and the generation block of the
other — the mechanism of `f1_vla_policy_meankv.py`.

`follow_forecast = d(AB, AA) / (d(AB, AA) + d(AB, BB))`: 0 = the action
stays with its instruction, 1 = it goes where the forecast's instruction
points.

## Instruction pair

Main pair, Stack scene: "stack the green block on the yellow block" vs.
"stack the yellow block on the green block". Both objects are in the scene,
both goals are physically possible, the two strings have the same tokens
(so the token count and all position ids are identical), and the simulator
already reports which cube was moved (`moved_correct_obj`,
`moved_wrong_obj`).

## Steps

0. **Strength check.** How much does the instruction change the 30 forecast
   tokens and the generation block's keys/values, compared with a change of
   sampling seed under the same instruction? If the instruction changes the
   forecast no more than the seed does, the swap is a weak intervention (the
   wrong-episode shuffle changed the block by 1–3%) and the result is "the
   forecast does not carry the instruction".
1. **Actions at identical states**, all four cells.
2. **Closed loop** for the conflict cells: which cube does the robot go for?

## Pitfalls

- Token count must match between A and B (language is padded to 48, and
  position ids count only real tokens).
- World-model sampling is stochastic: fix the generator, and report the
  seed-to-seed difference as the noise floor.
- The generation block also holds the history frames; they are the same in
  both runs, but their keys/values in deeper layers still depend on the
  instruction. Swap the whole block first, then future tokens only.
- The swapped instruction is rare or absent in Bridge training data; check
  first that under BB the policy actually goes for the yellow cube.
