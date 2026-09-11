# DDQN run analysis

Analysis date: 2026-09-11. Code inspected from this worktree at commit
`c66a424`. Run artifacts were read read-only from the original checkout at
`/home/dev/Projects/dodge-gymnasium/history/dodge/gymnasium/cnn-image-ddqn/`.

## Scope and evidence limits

There are 24 run directories (23 runs plus the shared `rewards.json`), including
four long CUDA runs and several smoke/probe runs. The artifact contract uses a
fixed training seed (42) and only four episodes per inner/holdout evaluation.
That is enough to compare the recorded checkpoints, but not enough to estimate
seed variance or claim a general improvement. Most long runs have a `warn` gate
because the configured one-million-step epsilon schedule was capped to the run
length; this is a schedule/provenance warning, not a measured failure of the
policy.

## Recorded long-run comparison

| run | steps | learning rate | warmup / target sync | inner | holdout | counterfactual | holdout gap | dead units | action balance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `diag-50k-t4-001` | 50k | 2.5e-4 | 32 / 100 | 101.0 | 101.0 | 101.0 | 0.000 | 0.341 | 0.842 |
| `fix-cfg-50k` | 50k | 2.5e-4 | 20k / 10k | 228.0 | 218.0 | 165.0 | 0.044 | 0.306 | 0.636 |
| `fix-plain-50k` | 50k | 2.5e-4 | 20k / 10k | 213.0 | 218.0 | 242.25 | 0.023 | 0.393 | 0.778 |
| `lr1e4-250k` | 250k | 1.0e-4 | 20k / 10k | 209.0 | 183.0 | 175.5 | 0.124 | 0.563 | 0.895 |
| `cfg-250k-t4-001` | 250k | 2.5e-4 | 20k / 10k | 142.5 | 130.75 | 101.0 | 0.082 | 0.786 | 0.929 |
| `diag-250k-t4-001` | 250k | 2.5e-4 | 32 / 100 | 196.5 | 196.0 | 182.5 | 0.003 | 0.500 | 0.964 |
| `diag-500k-t4-001` | 450k* | 2.5e-4 | 32 / 100 | 196.5 | 196.0 | 182.5 | 0.003 | 0.547 | 0.962 |
| `lr1e4-scale` | 500k* | 1.0e-4 | 20k / 10k | 254.25 | 254.5 | 162.0 | 0.001 | 0.579 | 0.891 |

`*` indicates a resumed run. The 450k run resumes at 50k; the 500k run
resumes at 250k. The resume histories make these rows useful for trajectory
inspection, but they are not independent trials.

## What is provable from the current artifacts

1. **The 1e-4 configuration is the best recorded checkpoint.** `lr1e4-scale`
   has 254.5 holdout reward versus 196.0 for the same broad 2.5e-4 diagnostic
   family. It also has a 0.001 normalized train/holdout gap. This is a strong
   candidate, not a causal estimate: it is a resumed run and has a different
   optimizer/history from the 2.5e-4 rows.

2. **More steps alone did not improve the 2.5e-4 diagnostic policy.** The
   250k and resumed 450k checkpoints both evaluate at 196.0 holdout reward,
   while the 50k checkpoint is 101.0. This suggests an early plateau or a
   policy attractor under that configuration; it does not prove that training
   longer is useless because the runs share seed and checkpoint lineage.

3. **The 20k warmup / 10k target-sync configuration is materially different,
   and its 50k result is better than the 32 / 100 diagnostic result.** The
   `fix-*` rows reach 218 holdout at 50k versus 101. The comparison changes two
   hyperparameters at once, so it identifies a promising bundle rather than
   assigning credit to warmup or target cadence separately.

4. **Reward sparsity is not the limiting signal in these long runs.** Final
   `reward_zero_share` is 0.002–0.012, well below the 0.95 sparse-feedback
   failure threshold. The observed failure modes are value/representation
   dynamics and policy coverage, not an all-zero reward stream.

5. **Action lock-in is not triggered by the gate, but action coverage is still
   informative.** Final balance is 0.636–0.964 in long runs (the gate only
   flags below 0.05). The counterfactual probe is decisive evidence that the
   least-used action can be either much worse (`fix-cfg`: 165 versus 228 inner)
   or better (`fix-plain`: 242.25 versus 213 inner; `lr1e4-scale`: 162 versus
   254.25), so action frequency alone must not be treated as policy quality.

6. **Dead-unit share remains high.** Long-run final values range from 0.30 to
   0.79. The 2.5e-4 `cfg` run has the worst representation probe (0.786) and
   the weakest holdout among comparable 250k runs. This is a measurable
   optimization target, but the probe is a trunk zero fraction, not proof that
   those units are permanently incapable of learning.

7. **The Q scale is configuration-dependent and can drift.** Final Q means are
   about 4 for `fix-*`, 23 for the 250k standard runs, 66 for `lr1e4-scale`,
   and 395 for `diag-*`. Loss values and Q magnitudes therefore cannot be
   compared across rows without normalizing by target/Q scale. In particular,
   a flat or small loss is not evidence of a good policy.

## Code structure relevant to performance

The variant has a clean separation: Rust owns transitions and collision images;
Python owns frame stacking, replay, the Atari-style 3-convolution dueling head,
the Double-DQN update, diagnostics, and artifact writing. The learner uses
Huber loss, Adam, gradient clipping at 10, and a hard target copy. Replay stores
owned `uint8` frames and converts to float32 at update time. These boundaries
make controlled learner/configuration experiments feasible without changing
game semantics.

The current model is approximately 1.7M parameters before the head and receives
84x84 images with 1 or 4 channels. The main structural risks visible in the
artifacts are dead ReLU activations, large Q-scale drift, and a training/eval
protocol too small to estimate variance. None requires changing the native
environment.

## Experiments that can establish causality

Run each row with at least five independent seeds, identical total steps,
identical evaluation seeds, and a locked holdout that is not used for selection.
Report mean, standard deviation, and confidence intervals for holdout reward and
survival, plus the existing dead-unit, Q-spread, TD-error, reward-zero, action
balance, and counterfactual metrics.

1. **Learning rate:** 1e-4 versus 2.5e-4, fresh checkpoints, 250k steps. This
   is the highest-value factor because the best observed row uses 1e-4, but the
   current evidence is confounded by resume state.
2. **Warmup:** 32 versus 20k, holding target sync at 100 (and then at 10k in a
   second factorial block). This separates replay age from target staleness.
3. **Target cadence:** 100, 1k, and 10k with warmup and learning rate fixed.
   Log the number of syncs and evaluate at equal optimizer steps.
4. **Representation:** 1-channel versus 4-channel stack at the winning
   optimizer settings. Keep the native image and action semantics unchanged.
5. **Dead-unit mitigation:** only after the above baseline is replicated,
   compare a less saturating activation or orthogonal/He initialization. Accept
   only if holdout improves without Q-scale divergence.

Promotion rule: a change is an improvement only when the seed-level holdout
   lower confidence bound beats the baseline, the train/holdout gap stays below
   0.5, no sparse-reward or lock-in gate fires, and the result survives a fresh
   locked evaluation. CPU smoke runs can screen wiring; they cannot establish
   GPU throughput or learning quality.

## Immediate recommendation

Do not tune reward weights or add game semantics. First reproduce a fresh,
independent 250k 1e-4 baseline and a matched 2.5e-4 control with the same
warmup/target schedule. Then run the warmup/target factorial. The existing
`lr1e4-scale` checkpoint is the candidate to beat, while `diag-250k-t4-001` is
the cleanest same-family control. Increase evaluation episodes before using
small reward differences as evidence.
