# Native-pixel reward experiments

All treatments use full native RGB, four temporal frames, 700 game seeds,
learner seed 42, 200k decisions, warmup 20k, learning rate 1e-4, target sync
10k optimizer updates, batch 32, update every four decisions, replay 100k,
and epsilon decay 500k. A retained 10k checkpoint is explicitly untrained.
Evaluation uses 128 frozen inner and 128 holdout episodes at base seed 512,
with a 4096-decision cap and unshaped survival reward.

| Treatment | Death | Pickup | Native destruction | Edge/frame | Corner/frame |
|---|---:|---:|---:|---:|---:|
| `rgbcontrol` / `survival-v1` | 0 | 0 | 0 | 0 | 0 |
| `rgbdeath` / `death-v1` | -25 | 0 | 0 | 0 | 0 |
| `rgbevents` / `events-v1` | 0 | +10 | +1 | 0 | 0 |
| `rgbboundary` / `boundary-v1` | 0 | 0 | 0 | -0.01 max | -0.1 max |

Every treatment retains +1 per native survival frame. The spatial terms use
the approved two-pixel edge band and 16-pixel corner overlap, sampled at the
decision endpoint and multiplied by actual native frames advanced. Values are
initial experimental weights, not demonstrated optima.

`EnemyDestroyed` counts increments of the native shattered counter, including
indirect destruction and powerup entities destroyed by explosions. A collected
explosive powerup can therefore also contribute a destruction event. Pickup
counts are exact collection events, not score residuals. The event treatment
tests the two bonuses together; individual pickup/destruction attribution
requires a subsequent ablation if it helps.

Release order: T4 lane A runs control then death; lane B runs boundary then
events. Each lane first runs three alternating minimal/routine 10k telemetry
pairs with the combined reward profile. Every pair must have at most 5% added
training-loop wall time before the lane releases its 200k screens. A failed
gate stops that lane. This measures incremental logging overhead with reward
computation enabled in both arms; it is not a before/after benchmark of the
entire native extension. All-in and training-loop timing remain separate.

The combined profile is used for that overhead gate only, not a 200k treatment.
No automatic 500k promotion is encoded here: promotion needs the completed
matched survival distributions, then an A/H allocation. The user authorized
launch without another approval once those checks pass.

Native component arrays never enter the model observation. Weights and native
term version are recorded in config, manifest, and checkpoints; incompatible
resume rejects. Reports/logs retain cumulative weighted components in order:
survival, death, pickups, native destruction, edge, corner. These totals are
per-decision rewards before discounting/n-step aggregation, not replay samples.
Legacy Python shaping and live controls cannot be combined with new profiles.

Correctness gate: 191 Python tests, 86 Rust tests, Ruff and workspace Clippy
passed before packaging. Native-RGB smoke tests exercised all three treatment
profiles, optimizer updates, checkpoint publication, and incompatible resume.
The legacy survival-only quality gate is unchanged; a `pass` alone does not
establish learning or justify promotion.

The first GPU preflight exposed repeated native Death events from overlapping
hazards in one frame. Release was stopped before any 200k treatment. Native
reward now normalizes death presence to one life penalty while retaining
pickup/destruction multiplicity; an actual overlapping-hazard fixture covers
the source behavior. Rejected preflight archives are retained outside the
normal run catalog under `history/dodge/gymnasium/rejected-campaigns/`.
The corrected build reruns every telemetry pair. The campaign collector now
also preserves a supervisor-published failure archive before releasing a failed
assignment, so a failed gate does not leave an idle GPU waiting for runs that
were never released.
