# P7 — Controller comparison design (APPROVED 2026-09-18)

Authority: [SPEC](../SPEC.md) C9–C11, V10–V13, §E; card [07](phases/07-controller-design.md).
Predecessor state: P6 satisfied per SPEC P6.record (AD4–AD10 evidence). Owner
approved this design with the §6 powerup amendment: no aborts on gameplay
colors; policies face powerups and learn them. P8 opens for DDQN first.

## 1. Shared frozen core (all branches)

- World: `scale-resume-20000` (`4d8a5238`), frozen. Encoder + projector only;
  predictor unused except by the planning/hybrid branches (read-only).
- Policy state: 3 most recent projected 192-dim latents (576-dim), same history
  the v12 planner used. No pixels, no actions-in-state, no probe scores as
  features (the probe is a planning cost, not a policy input).
- Actions: native 9, legality by construction (categorical head / 9 logits).
- Cadence: 4 native frames per decision, unchanged.

## 2. Reward (pinned, unshaped)

Native `RewardTerms` with weights `[survival 1, death 1, pickups 0,
enemy_deaths 0, edge 0, corner 0]` per decision: +frames survived, −1 on the
decision containing death. Rejected: pickups/enemy-deaths (luck-driven score,
legacy `rewards.py` strips these for the same reason), edge/corner (shaping,
forbidden by C2), survival-only without death (identical ranking to
survival+death on fixed-horizon episodes, but death grounds the terminal TD
target). This matches the MPC survival-frames metric exactly, so planner and
policy numbers are directly comparable. Termination/truncation native.

## 3. Branches

### 3a. Planning reference (no training)

Frozen v12 `mpc_h3` (mean cost, H=3) on the frozen 20k world + frozen v12 probe.
Reference numbers already in hand (pooled +17.3 [+10.0,+24.6] vs random).
Reused as the beat-this baseline; no refit, no new search.

### 3b. DDQN (off-policy)

- Head: 576→256→256→9 dueling MLP on frozen latents. Target net separate (V10).
- Data: replay filled from epsilon-greedy rollouts in the 64 eval scenarios +
  mortal probe-set episodes; world-model pretraining corpus counts toward the
  branch's transition budget per V11 (equal exposure incl. corpus).
- Objective: 1-step TD on the §2 reward, frozen latents (no encoder grads).

### 3c. PPO (on-policy)

- Head: shared 576→256→256 trunk, 9-logit policy + scalar value heads.
- Data: fresh on-policy rollouts only (V11 forbids relabeling replay as
  on-policy). Rollout transitions count toward the same budget as DDQN's.
- Objective: clipped surrogate + value MSE + entropy bonus (coefficient pinned
  before training; entropy is an optimizer term, not reward shaping).

### 3d. Hybrid (planning + learned terminal value)

- Planner: v12 H3 search, sequence cost = mean probe cost + γ³·V(ẑ_terminal),
  where V is a 192→64→1 MLP fitted by TD on frozen latents from the DDQN
  branch's replay (same data, no new collection).
- Tests whether a learned terminal value beats the probe-only cost that won v12.

## 4. Fairness accounting (V11)

- One transition budget for all branches (proposed: 200k post-pretraining
  decisions each — owner to confirm), counted identically: replay insertions
  (DDQN), rollout steps (PPO), zero (planning reference, but its pretraining
  corpus is shared and equal by construction).
- Compute reported separately: pretraining (sunk, shared) vs controller
  training vs decision latency (mean ms/decision on the eval device).
- Same 64 eval scenarios for screening; confirmation on fresh seeds (§5).

## 5. Seeds, splits, gates

- Screen: the 64 AD10 scenarios (paired-valid vs v12 numbers).
- Confirm (P10): ≥5 fresh matched learner seeds + one sealed holdout opening
  (V12/V23); holdout scenarios generated but never touched until then.
- Gate per branch: V13 — positive 95% paired CI vs the strongest predeclared
  baseline (random at minimum; h3 reference for a superiority claim) plus
  scene-sensitive greedy behavior (argmin audit as in v11/v12).
- Advance rule: a branch that fails V13 at screen may not proceed to P10 on
  more seeds; negative/inconclusive is a reportable result (V22).

## 6. The powerup-sparkle rule (DECIDED: face powerups, no aborts)

Rust-verified facts: all enemy bodies incl. powerups render in-palette
(ENTITY_COLOR 7 cream + SHADOW_COLOR 1 navy); only ambient sparkles of
personality 2 ([8,9,10] red/orange/yellow) and personality 3 (index 13
lavender) fall outside the 3-color training palette. Practice-mode training
corpora suppress organic spawns, so sparkles are structurally train-absent.

Pinned rule (owner 2026-09-18): powerups stay ON; policies must learn them.

- Live-input boundary (`_history_batch` path, shared by planning rollouts,
  policy rollouts, and eval): exact-black keeps the AD6 mask-to-background
  (viewport artifact); every other out-of-palette pixel projects to the
  nearest training-palette color (RGB Euclidean, ties → lowest palette
  index, deterministic). Both counts (`masked_pixels`, `projected_pixels`)
  reported per episode. In-palette pixels pass bit-identical, so all prior
  probe/planning numbers remain comparable.
- The abort path stays as a fail-closed backstop but must never trigger on
  PICO-8 colors; any abort is a defect, not data.
- Frozen banks/datasets keep strict coverage validation (data must be
  exactly in-palette); projection applies to live inputs only.
- Corpus expansion with powerup personalities proceeds in parallel as the
  principled fix for the next world; it does not block P7.

## 7. Branch order and budgets (CONFIRMED)

- Order: DDQN → PPO → hybrid (hybrid reuses DDQN's replay + value).
- Caps per branch: 200k decisions, one T4-class worker, 7200s phases;
  PPO minibatch/epoch settings pinned at P8. These replace the retired DDQN
  campaign settings (§E).
- P8 implements ONE branch at a time (card 08); P9 screens it; P10 confirms
  only branches that pass the screen gate.

## 8. Approval record

Owner approved 2026-09-18: §6 powerup amendment (face powerups, projection
rule, no aborts), §7 budgets/order as proposed, P6 rationale as recorded in
SPEC P6.record. P8 opens for DDQN. Training still needs each phase's frozen
protocol + weights.
