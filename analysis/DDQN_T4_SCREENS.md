# Independent T4 screens

The completed 700-game-seed collision run supplies the baseline configuration:
learning rate 1e-4, gamma 0.99, dueling CNN, four frames, 100k replay capacity,
batch 32, warmup 20k decisions, target refresh every 10k optimizer updates,
one optimizer update per four decisions, and epsilon decay over 500k decisions.
Native game settings and rewards stay unchanged.

| Run | Change from baseline | Budget |
|---|---|---:|
| `t4-rep43-200k-v1` | Learner RNG seed 43 | 200k decisions |
| `t4-rep44-200k-v1` | Learner RNG seed 44 | 200k decisions |
| `t4-nstep3-200k-v1` | Three-step discounted returns | 200k decisions |
| `t4-gray-200k-v1` | Full native 128x128 grayscale instead of 84x84 collision raster | 200k decisions |

All start fresh and cycle through the same fixed 700-game-seed pool. Each saves
a 10k checkpoint, which is explicitly untrained because warmup lasts 20k.
The exploration schedule is not compressed to the screening budget. Compare
against the baseline's 200k checkpoint, not its 500k result.

Grayscale is an observation-representation experiment: it changes image content,
resolution, and the resulting CNN flatten size. It does not isolate color alone.
A later matched native RGB-versus-grayscale comparison can isolate color.
Three-step returns are currently restricted to the collision profile; terminal
suffixes stop bootstrap, truncation suffixes retain it, and segment-end suffixes
are flushed. Warmup remains measured in environment decisions for this branch.

Evaluation uses 128 inner and 128 holdout scenarios, seed base 512, disjoint
offsets, and a 4,096-decision cap. Snapshot evaluation runs only after training.
The original learner-42 baseline must be evaluated on this same scenario set
before claiming a matched improvement. Mean survival, early-death rates,
distribution spread, and censoring take precedence over a selected high score.

## Deployment

Colab admitted two additional T4 assignments; a third was rejected by the
concurrent-assignment limit. The A100 RGB target-refresh campaign is unchanged.

- `dodge-t4-replicates-v1`: seed 43, then seed 44. Frozen learner source `e7afc03`.
- `dodge-t4-nstep-v1`: three-step returns, then grayscale, after local tests and
  per-treatment GPU smoke checks pass.
- Each session logs to `/content/t4-campaign.log` and writes `/content/t4-history`.
- Replica collector: `/tmp/ddqn-t4-replicates-collector.log`.

Collectors download completed archives into the local variant history and
release only their exact assignment after all expected runs are collected.
Screening does not automatically promote a treatment to 500k or establish
replicated performance. No learning improvement is claimed at deployment.
