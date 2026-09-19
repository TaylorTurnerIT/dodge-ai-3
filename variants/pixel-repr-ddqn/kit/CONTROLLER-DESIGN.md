# P7 — Controller comparison design (DRAFT, unapproved)

Authority: [SPEC](../SPEC.md) C9–C11, V10–V13, §E; card [07](phases/07-controller-design.md).
Predecessor state: P6 formally unopened; AD4–AD10 supply its evidentiary content
(latent dynamics 0.95→0.13, V13-positive planning on the 20k world). This draft
scopes the comparison only. No branch opens, no code, no training until the
owner approves this design AND records how P6 is satisfied.

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

## 6. The powerup-sparkle problem (open, must resolve before P8)

Organic rollouts with powerups enabled contain red sparkles the frozen encoder
never saw (v12 abort). Branches need a pinned rule, options:

- (a) Powerups ON in training/rollouts, strict palette contract at eval
  (aborts stay loud; policies must cope with encoder degradation on sparkle
  frames — measures robustness, risks noise).
- (b) Powerups OFF everywhere in P7 (clean comparison, but changes the game
  vs v12 and dodges the gap instead of closing it).
- (c) Corpus expansion first: determined scenarios with powerup personalities
  (AD.data follow-up), encoder stays frozen but eval coverage is then honest —
  delays P7 by one collection phase.

Recommendation: (a) for the screen (honest, measures the real artifact), (c)
in parallel as the principled fix. Owner decides.

## 7. Branch order and budgets (owner to confirm)

- Order: DDQN → PPO → hybrid (hybrid reuses DDQN's replay + value).
- Proposed caps per branch: 200k decisions, one T4-class worker, 7200s phases;
  PPO minibatch/epoch settings pinned at P8. These replace the retired DDQN
  campaign settings (§E) once approved.
- P8 implements ONE branch at a time (card 08); P9 screens it; P10 confirms
  only branches that pass the screen gate.

## 8. What approval unlocks

Owner approval of this doc (as-is or amended) authorizes: (1) recording the
P6-satisfaction rationale, (2) amending P8/P9/P10 branch contracts per card
step 3, (3) opening P8 for the first branch. It does not authorize training —
each training phase still needs its frozen protocol + weights.
