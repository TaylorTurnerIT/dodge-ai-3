# Repeated-action probes and stronger-boundary experiments

## Measured offline

Final frozen checkpoints, 16 matched inner seeds (10512–10527), 512-decision cap,
native collision-image inputs, four game frames per action. No training process
was instrumented or changed. Exploration uses epsilon 0.64 with fixed RNG seeds;
it is a checkpoint behavior comparison, not reconstructed training experience.

| Checkpoint | Greedy mean frames | Exploratory mean frames |
|---|---:|---:|
| Original best, three-step seed 43 | 448.875 | 289.0 |
| Same configuration with boundary penalties | 173.6875 | 281.6875 |

For the boundary policy under greedy play, ordinary transitions had mean absolute
TD error 0.675. Transitions whose three-step window reached death had mean absolute
TD error 51.812. All 48 terminal-window samples were in the linear part of Smooth
L1. These are offline target-network errors, not measurements from training replay
batches or parameter-gradient norms. Large terminal errors also occurred in the
original best policy, so this signal alone does not explain the performance gap.

Exploration helps the poor boundary policy on these seeds but hurts the original
best. This supports testing whether random movement masks the poor policy's
failure during training. It does not establish a general causal result.

Full measurements: `DDQN_BOUNDARY_FAILURE_PROBE.json`, `DDQN_BEST_FAILURE_PROBE.json`.

## Experiments queued

Both start fresh for 200k steps on a T4. Reference:
`t4-nstep3s43boundary-200k-v1`. Preserve seed 43, 700 training seeds, collision
inputs, three-step returns, 10,000-optimizer-update target sync, epsilon schedule,
and frozen 128-seed evaluation. Change only boundary weights; no new death or
event rewards. Native Rust geometry and reward components remain unchanged.

| Run | Edge weight | Corner weight |
|---|---:|---:|
| Existing reference | 0.01 | 0.1 |
| t4-boundary5-200k-v1 | 0.05 | 0.5 |
| t4-boundary10-200k-v1 | 0.1 | 1.0 |

The native geometry returns edge=1 and corner=1 at an exact corner. Thus the
10x treatment can make ongoing deep-corner training reward negative. This is an
intentional stronger-penalty screen, not a presumed improvement: compare unshaped
survival and deaths to detect worse behavior rather than celebrating lower loss.

`scripts/ddqn-strong-boundary-queue.py` waits for a free T4 without interrupting
the target-sync sweep, then runs smoke checks and both treatments sequentially.
It collects both completed archives before releasing its own assignment. Queued
dashboard entries are explicitly planned and are replaced by native run artifacts
when training starts. Release launcher and reward weights are frozen in
`/tmp/ddqn-strong-release/`; runtime artifacts record their hashes.

Validation: 179 variant tests passed; repository Ruff check passed. No additional
per-step diagnostic work was added to training.
