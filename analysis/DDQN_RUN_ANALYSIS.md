# DDQN performance analysis

Analysis date: 2026-09-11. Code was inspected at `c66a424`. Artifacts were
read without modification from
`/home/dev/Projects/dodge-gymnasium/history/dodge/gymnasium/cnn-image-ddqn/`.

## Bottom line

No DDQN change is yet proven to improve policy quality. Every substantive run
uses training seed 42, most comparisons change more than one factor, evaluation
uses only four inner and four holdout episodes, and the same evaluation seeds
have already been used to compare configurations.

The best recorded final policy is `lr1e4-scale`: 254.5 mean holdout reward.
That is a candidate, not a causal learning-rate result. It inherited a 250k
checkpoint and then ran another 500k environment steps, so its true exposure is
750k steps. Resume restored the networks and optimizer but not replay, RNG,
environment state, or the exploration schedule. Its evaluation is also nearly
saturated at the 256-frame limit: five of eight inner/holdout episodes hit the
cap.

The current evidence supports four immediate corrections before further model
tuning:

1. make resume and schedule units explicit and comparable;
2. evaluate longer, over more training and evaluation seeds;
3. repair metrics that currently hide target-sync and policy-coverage failures;
4. run a fresh, matched `1e-4` versus `2.5e-4` experiment.

Reward sparsity is not the problem in the instrumented long runs. The final
512-transition windows have `reward_zero_share` between 0.002 and 0.012. The
native reward is survival frames advanced, normally four per decision, so a
continuing-state Q-value near `4 / (1 - 0.99) = 400` is within the reward-scale
ceiling and is not, by itself, proof of overestimation.

## Evidence inventory and limits

The artifact root contains 25 status-bearing runs: 22 completed, two stopped,
and one failed. Eight completed runs have the newer diagnostics and are long
enough to compare trajectories. Two earlier `he-init-*` CUDA runs use the old
artifact schema and exploration behavior; they have no frozen holdout,
counterfactual, Q, TD-error, action-balance, or activation metrics.

The manifests record the variant, device, seed, observation version, and resume
path, but not the Git commit, dirty state, dependency lock hash, initialization
scheme, checkpoint hash, or parent-run ID. Run names such as `he-init-*` are not
machine-verifiable provenance. Conclusions below therefore use only factors
present in `config.json`, `manifest.json`, and the measured artifacts.

All substantive runs use:

- one serial environment lane and training seed 42;
- a 4x84x84 collision-image stack and nine actions;
- batch size 32, replay capacity 100k, update every four decisions;
- `gamma=0.99`, Huber loss, Adam, and gradient clipping configured at 10;
- four inner and four holdout episodes, each capped at 64 decisions or at most
  256 native survival frames.

The evaluation reward and survival-frame fields differ by approximately one at
termination and otherwise encode the same outcome. They are not independent
quality measures.

Key code evidence:

- `model.py:70-130` defines the CNN, initialization, and dueling combine;
- `agent.py:153-205` defines the Double-DQN update and available diagnostics;
- `run.py:904-982` defines collection, replay warmup, updates, and target sync;
- `run.py:1013-1087` defines the training metric denominators;
- `run.py:1124-1243` defines evaluation, gates, and final artifacts;
- `diagnostics.py:82-97` defines cumulative action balance;
- `replay.py:94-100` shows the two full-stack replay allocations;
- `native/crates/dodge-batch/src/collision_image.rs:31-54` defines the
  max-composite observation;
- `native/crates/dodge-batch/src/lib.rs:1995-2014` defines reward as survival
  frames advanced.

## Run comparison

`eval mean` and `eval SD` pool the eight greedy inner/holdout episodes. The SD
is episode dispersion inside one trained seed, not run-to-run uncertainty.
`syncs` is inferred from optimizer steps and the configured optimizer-update
interval.

| run | total env steps | LR | warmup | target interval | updates / syncs | eval mean +/- SD | cap share | final eval activation zeros |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `diag-50k-t4-001` | 50k | 2.5e-4 | 32 | 100 | 12,493 / 124 | 101.0 +/- 0.0 | 0% | 0.343 |
| `fix-cfg-50k` | 50k | 2.5e-4 | 20k | 10k | 7,501 / **0** | 223.0 +/- 27.9 | 37.5% | 0.382 |
| `fix-plain-50k` | 50k | 2.5e-4 | 20k | 10k | 7,501 / **0** | 215.5 +/- 40.1 | 37.5% | 0.295 |
| `cfg-250k-t4-001` | 250k | 2.5e-4 | 20k | 10k | 57,501 / 5 | 136.6 +/- 21.2 | 0% | 0.785 |
| `lr1e4-250k` | 250k | 1e-4 | 20k | 10k | 57,501 / 5 | 196.0 +/- 41.6 | 25% | 0.558 |
| `diag-250k-t4-001` | 250k | 2.5e-4 | 32 | 100 | 62,493 / 624 | 196.3 +/- 0.7 | 0% | 0.500 |
| `diag-500k-t4-001` | 500k* | 2.5e-4 | 32 | 100 | 124,986 / 1,249 | 196.3 +/- 0.7 | 0% | 0.547 |
| `lr1e4-scale` | **750k*** | 1e-4 | 20k | 10k | 177,502 / 17 | 254.4 +/- 2.7 | 62.5% | 0.570 |

`*` includes the parent checkpoint's recorded steps. `diag-500k-t4-001` loads a
50k checkpoint and runs 450k new steps. `lr1e4-scale` loads a 250k checkpoint
and runs 500k new steps.

The older `he-init-50k-t4-001` and resumed `he-init-500k-t4-001` rows evaluate
at 88.0 and 196.5 on four inner episodes. Their epsilon values end at 0.955 and
0.595 because they used the uncapped one-million-step schedule. They are not
comparable to the later runs, which force epsilon to reach 0.1 at the end of
every segment.

## Network and learner structure

The current network is a conventional Atari CNN:

| stage | output | parameters |
|---|---:|---:|
| Conv 8x8/4, 4 -> 32 + ReLU | 32x20x20 | 8,224 |
| Conv 4x4/2, 32 -> 64 + ReLU | 64x9x9 | 32,832 |
| Conv 3x3/1, 64 -> 64 + ReLU | 64x7x7 | 36,928 |
| Linear 3,136 -> 512 + ReLU | 512 | 1,606,144 |
| Dueling value + advantage heads | 1 and 9 | 5,130 |
| **total** | 9 Q-values | **1,689,258** |

The plain head removes the value stream and has 1,688,745 parameters, only 513
fewer. The dense 3,136-to-512 layer holds 95.1% of the dueling model's
parameters. Any capacity or regularization experiment should focus there
rather than on the small output heads.

Current initialization is Kaiming uniform for ReLU layers and Xavier uniform
for the linear Q heads. That is structurally appropriate. The artifacts do not
record initialization identity, so the existing runs cannot prove that this
change helped.

The collision image is a one-channel max-composite raster: background 0,
damaging area 96, hostile 192, player 255. It preserves collision geometry but
collapses hostile subtypes and other latent game state. Four frames provide
recent motion but no explicit velocity, time-to-contact, pattern phase, or
spawn intent. This partial observability is a plausible representation limit,
not a demonstrated failure cause.

The Double-DQN target is implemented correctly: the online network selects the
next action and the target network evaluates it. A learner update performs
three batch forwards (online current, online next, target next), one backward
pass, and one optimizer step. Action collection also performs a batch-of-one
CNN forward on every environment decision.

## Variance and trajectory findings

### Seed variance is unknown

There is no between-training-seed estimate. Repeating deterministic evaluation
seeds on one checkpoint measures scenario dispersion, not training stability.
No confidence interval over the current rows can support a general improvement
claim.

### Evaluation variance is censored

The strongest policy, `lr1e4-scale`, has inner/holdout rewards from 250 to 256;
five of eight episodes hit 256. Its low SD therefore reflects the evaluation
ceiling as well as consistency. Longer evaluation is required to rank it
against policies that also survive 64 decisions.

The 50k fixed-schedule runs have much larger within-checkpoint dispersion:
27.9 for dueling and 40.1 for plain. Both have holdout mean 218.0, so this
single-seed result supplies no evidence that the dueling head helps.

### Learner variance differs by schedule

For each rich run, the table below compares medians over the first and final
10% of logged post-warmup rows. These are correlated trajectory samples, not
replicates.

| run | Q mean early -> late | TD-error SD early -> late | late loss median / p90 | activation-zero share early -> late |
|---|---:|---:|---:|---:|
| `diag-50k-t4-001` | 27.2 -> 266.6 | 0.86 -> 2.65 | 3.02 / 17.33 | 0.328 -> 0.348 |
| `fix-cfg-50k` | 4.09 -> 4.07 | 0.19 -> 0.13 | 0.010 / 0.088 | 0.338 -> 0.403 |
| `fix-plain-50k` | 3.99 -> 4.00 | 0.21 -> 0.14 | 0.010 / 0.102 | 0.293 -> 0.307 |
| `cfg-250k-t4-001` | 4.09 -> 23.34 | 0.14 -> 0.27 | 0.037 / 0.811 | 0.409 -> 0.779 |
| `lr1e4-250k` | 4.10 -> 23.40 | 0.15 -> 0.17 | 0.019 / 0.725 | 0.338 -> 0.517 |
| `diag-250k-t4-001` | 114.8 -> 399.4 | 1.68 -> 69.47 | 12.49 / 25.34 | 0.315 -> 0.500 |
| `diag-500k-t4-001` | 320.7 -> 407.9 | 2.41 -> 69.55 | 12.49 / 25.87 | 0.352 -> 0.547 |
| `lr1e4-scale` | 27.1 -> 66.0 | 0.18 -> 0.33 | 0.065 / 2.24 | 0.522 -> 0.576 |

The 100-update target schedule drives Q estimates toward the approximate
continuing-return ceiling and produces large, spiky TD errors. The 10k-update
schedule yields smaller Q and TD scales but leaves the target network stale for
long stretches. These are measured associations; warmup, target cadence,
epsilon schedule, resume state, and learning rate are confounded across
families.

Final checkpoint weights confirm the target-lag difference. Online-to-target
relative L2 distance is 0.001-0.031 in the 100-update runs, versus 0.106-0.347
in the 10k-update runs. The two 50k fixed runs never copy the target at all, so
their target remains the initial network even though the quality gate does not
report `target-never-synced`.

The logged `dead_units` field is not a dead-neuron measurement. It is the exact
zero fraction across the final convolutional activation tensor for one current
observation; evaluation averages the same fraction across visited states. It
does not identify units that are zero for all inputs and excludes the 512-wide
shared ReLU. Across these eight runs, the descriptive correlation between
pooled evaluation reward and this fraction is -0.10, so the current evidence
does not show that reducing it improves policy reward.

## Metric audit

### Reliable for the stated scope

- Per-evaluation seed rewards and survival frames are deterministic checkpoint
  outcomes, subject to the short horizon.
- `optimizer_step`, replay size, epsilon, and cumulative action counts expose
  the actual loop state.
- Q/target/TD standard deviations and loss are valid minibatch snapshots.
- `reward_zero_share` correctly describes the last 512 collected transitions.
- Throughput is environment decisions per training-loop second. It includes
  collection and learning, but excludes final evaluation and artifact
  finalization. It is not native frames per second.

### Misleading or incomplete

- `target_sync_interval` is applied to optimizer steps, but `decide_gate`
  compares it with environment steps. This misses the zero-sync 50k runs.
- Epsilon decay is capped independently to each run segment. Runs of different
  lengths therefore change both training duration and exploration rate; resume
  restarts epsilon at 1.0.
- Resume restores model, target, optimizer, and optimizer-step count, but starts
  empty replay, reseeds RNGs, resets the environment, reruns warmup, and treats
  the new segment's step count as local. `lr1e4-scale` is consequently a 750k
  exposure with a distribution reset.
- `action_balance` ignores zero-count actions in its min/max ratio and is
  cumulative from step one. With epsilon initially near 1, a high value mostly
  confirms random exploration, not balanced greedy behavior.
- The counterfactual uses offset-30k seeds, while baseline evaluation uses
  offset-10k and offset-20k seeds. Differences cannot be attributed to the
  forced first action because the scenarios are not paired.
- Logged training `reward` is whichever episode total exists at the logging
  instant. It is often a partial episode, and completed episodes between log
  boundaries are omitted. `best_score` can therefore miss the true best
  training episode. Neither is a valid episode-return series.
- `mean_target` and pre-clip gradient norm exist in `DDQNUpdate` but are not
  written to metrics. Clipping frequency, target bias, and Q-target calibration
  cannot be recovered.
- `q_std` during training is over chosen actions in one replay minibatch;
  evaluation `q_std` pools all actions and visited states. They have different
  denominators.
- The gate warns every rich run for `epsilon-decay-scaled`, but this provenance
  warning is mixed with behavioral warnings. No current run passes.

## Changes that improve what can be proved

These changes improve experimental validity directly, regardless of whether a
policy score rises:

1. Record Git commit and dirty state, dependency-lock hash, initialization ID,
   parent run/checkpoint hash, global environment step, and global optimizer
   step in every artifact.
2. Define epsilon and target schedules in named units. On resume, either restore
   replay/RNG/environment/schedule state or label the operation as a new phase
   with an exploration and replay reset. Never report local segment steps as
   total exposure.
3. Make the target-sync gate use optimizer steps and record sync count plus
   online-target distance. A run with zero post-initialization copies must be
   visible.
4. Evaluate to termination or use a horizon well above observed survival.
   Increase evaluation episodes, retain per-seed rows, and keep one untouched
   locked seed set for the final decision.
5. Pair counterfactual and baseline rollouts on the same seeds. Report paired
   reward deltas rather than comparing different seed offsets.
6. Log complete episode returns at termination. Separately log rolling windows
   for recent behavior, greedy-action counts, terminal share, replay age, target
   mean, Q-minus-target mean, pre/post-clip gradient norm, clipping frequency,
   and per-layer activation occupancy.
7. Count missing actions as zero in balance metrics and report recent greedy
   policy balance after exploration has fallen, not only cumulative behavior
   actions.

## Experiments that can prove a learning improvement

Use at least five independent training seeds per cell, paired evaluation seeds,
equal global environment and optimizer steps, the same uncensored evaluation
horizon, and a locked final test set. Report the seed-level mean, standard
deviation, bootstrap confidence interval, and paired effect versus control.
Predeclare the primary endpoint as mean survival frames; treat Q scale, TD
error, action coverage, and activation occupancy as diagnostics rather than
success criteria.

Run these in order:

1. **Learning rate:** fresh `1e-4` versus `2.5e-4`, with identical warmup,
   target cadence, and fixed global epsilon schedule. This tests the most
   promising recorded difference without the 750k/resume confound.
2. **Target lag:** compare 100, 1k, and 10k optimizer updates at the winning
   learning rate. Evaluate at equal optimizer counts and record actual syncs.
   The existing results show materially different value dynamics, not which
   lag is best.
3. **Warmup:** compare 32 versus 20k while holding target lag fixed. The current
   50k rows change both together and cannot assign credit.
4. **Head:** plain versus dueling under the winning optimizer settings. The
   matched 50k single-seed rows have identical holdout means, so dueling has not
   earned its place yet.
5. **Observation:** only after the optimizer baseline is stable, compare four
   frames with a native-owned semantic or hazard representation. Preserve the
   same action, reward, and termination semantics. This tests partial
   observability without confounding reward design.

Do not tune reward weights from these artifacts. The reward is already dense,
and changing it would alter the objective before optimizer and evaluation
confounds are resolved.

## Wall-clock performance opportunities

The replay buffer allocates both current and next 4x84x84 uint8 stacks for
100k transitions: 5,644,800,000 bytes (5.26 GiB) before small action/reward/done
arrays. Consecutive stacks overlap heavily. Storing individual frames once and
reconstructing stacks at sample time can reduce image storage toward 0.66 GiB,
subject to episode-boundary handling and parity tests.

The single environment lane also feeds a batch-of-one GPU inference on every
decision, while every learner update performs three batch forwards. Batched
native lanes, batched action selection, pinned replay batches, and avoiding
redundant full-stack copies are the clearest throughput candidates. Benchmark
them separately from learning changes with identical transition counts and
report environment decisions/s, native frames/s, learner updates/s, transfer
time, and end-to-end wall time. The recorded 230-296 decisions/s values are
all-in training-loop snapshots, not controlled throughput trials.

## Recommended next run

First fix the schedule/resume counters, target-sync gate, complete-episode
logging, and uncensored evaluation. Then run a fresh factorial with learning
rate `{1e-4, 2.5e-4}` and target interval `{100, 1k, 10k}` optimizer updates,
holding warmup at 20k and using at least five training seeds. Promote a setting
only if its locked-test paired confidence interval beats the baseline and no
stability gate regresses.
