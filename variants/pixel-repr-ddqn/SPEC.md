# LeWM foundation and deferred controller comparisons

Status: MVP engineering delivery complete; scientific representation/controller phases remain unopened. Date: 2026-09-14.
Variant: `pixel-repr-ddqn`. Source namespace: `pixel_repr_ddqn`.
Worktree branch: `experiments/pixel-repr-ddqn`; base `460e662`.
Derived from prior conversation plan, SHA256 `0adc53c7e22036822d8b46fd4e23182f4c6aed81991c54df1178510fdbf9b3f5`. This file is self-contained; earlier worktree is not a runtime or documentation dependency.
Root FORMAT.md absent; existing compact SPEC/table convention retained. User explicitly requested phase kit; this spec defines its authority before kit creation.

Architecture amendment: LeWM replaces masked teacher/student + inverse/reconstruction multitask design. Existing filesystem slug retained for continuity; DDQN no longer mandatory. Earlier design archived in [reference snapshot](archive/PRE-LEWM-SPEC.md).

§G
G1|Learn stable action-conditioned LeWM dynamics from pixels/actions.
G2|Validate retained small-object and temporal information without privileged labels.
G3|Later compare planning, learned policies including DDQN/PPO, and planning+value; no predetermined winning controller.
G4|Keep world-model learning, diagnostic fitting, controller implementation and controller training separate.

§C
C1|Representation inputs/targets: native RGB pixels + executed actions only. Native positions/velocities/entity labels/collision masks/TTC/lookahead/privileged teachers forbidden. Timestamp/lane/episode metadata only sampling/masking/provenance.
C2|Ordinary native reward/termination/truncation allowed for RL targets and boundaries; no new game semantics/reward shaping. Rust remains sole authority.
C3|Store native128 RGB; nine actions; fixed4-native-frame decisions. Initial reference predictor history3, four observations per training window. Preprocessing to reference224 locked P0; no future information at action selection.
C4|LeWM learns encoder/predictor jointly from pixels/actions; no EMA teacher, stop-gradient prediction targets, external weights, inverse loss or reconstruction loss. User screenshot/taxonomy contextual only.
C5|New variant `pixel-repr-ddqn`; source `src/dodge_native_game/variants/pixel_repr_ddqn/`; tests `tests/variant_pixel_repr_ddqn/`; artifacts `history/dodge/gymnasium/pixel-repr-ddqn/`. No writes to legacy run roots.
C6|Own scope authorized by user; prior CNN-namespace-only constraint applies to legacy variant, not this new variant. Legacy code/spec/config/checkpoints unchanged except additive root routing. Native helper reuse through explicit audited imports; any shared extraction requires separate reviewed spec amendment and parity evidence.
C7|User2026-09-14 authorizes MVP implementation, Luna agents at max reasoning plus parent review, local upstream/paper references, Colab T4 model/decoder runs, bounded end-to-end verification, no-scroll dashboard and Tailscale exposure. No full research campaign or controller training authorized.
C8|Coding phases and collection/training phases separate. Coding phases permit only capped verification; training phases consume frozen code/protocol/data. No opportunistic future-phase work.
C9|LeWM foundation first. Deferred comparison families: planning; learned policies (DDQN and PPO); planning plus learned terminal value. PPO is the newer alternative named by user; no additional unspecified algorithm. No controller implementation until accepted P6 and scoped P7 design.
C10|SPEC owns behavior, interfaces, phase state, budgets, invariants and tasks. Kit owns execution order/templates and cites SPEC; no competing acceptance rules. Root historical board/PPO tasks do not select work for this variant.
C11|Numbers are proposed experiment settings, not established optimal choices. P0 fixes protocol choices; P3 calibrates/locks numeric controls before P4 training. Later changes invalidate affected comparisons and require versioned protocol amendment.

C12|User authorizes config-driven native testing scenarios in isolated worktree: enemies all/none/normal; patterns normal/off/permanent; easy/medium/hard; optional player invulnerability. Native/shared edits limited to config plumbing, rule enforcement and snapshot preservation; default behavior/wire8 unchanged. No new controller/training campaign.

§R
R1|LeWM v3|Joint pixel/action latent prediction + SIGReg; no EMA/stop-gradient/pretrained encoder/reconstruction objective|https://arxiv.org/html/2603.19312v3
R2|Training reference|Targets remain attached; SIGReg on time×batch×dimension; one-step teacher-forced loss|https://github.com/lucas-maes/le-wm/blob/main/train.py
R3|Model reference|CLS frame representation, causal action-conditioned predictor, separate BatchNorm projectors|https://github.com/lucas-maes/le-wm/blob/main/config/train/model/lewm.yaml
R4|SIGReg reference|Random direction projections; statistic averages across batch at each time position, then over projections/time|https://github.com/lucas-maes/le-wm/blob/main/module.py
R5|Control boundary|Paper uses goal-image MPC with continuous actions; Dodge survival and nine discrete actions require later explicit cost/controller design|https://arxiv.org/html/2603.19312v3#S3.SS2
R6|Uncertainty|No novelty claim for LeWM+DDQN/PPO without separate prior-art review. Noncollapse does not establish small-hazard retention or survival usefulness.

§I
pixels: `native_adapter.py` → owned native RGB/action whitelist; preprocess explicit, no privileged features.
dataset: `dataset.py` → same-lane episode-contained history/future windows, train/validation/inner/final-holdout isolation, raw-pixel hashes.
collector: `collect.py` → fixed exploratory policies; no optimizer; versioned corpus in own artifact root.
world_model: `model.py` → encoder E, action embedder U, causal predictor F, projectors; encode/history/rollout interfaces versioned.
regularizer: `sigreg.py` → time×batch×latent statistic; random projection RNG and all buffers checkpointed.
pretrain: `pretrain.py` → prediction + SIGReg only; all prediction branches receive gradients; code/data/protocol frozen in training phases.
probes: `probes.py` → frozen-model diagnostics, train-only fitted visual decoder/readouts, held-out open-loop evaluation; gradients never update LeWM.
controllers: deferred common interface → recent image/action history, frozen model identity, selected discrete action, timing; DDQN/PPO/planner/hybrid own later protocols.
artifacts: `run_artifacts.py` → own variant root; preserve familiar manifest/config/status/metrics/evaluation/report names; no legacy path authority.
checkpoint: encoder/predictor/action/projectors, BatchNorm buffers, optimizer/scheduler/RNG, data position, global counters, provenance; no EMA fields implied.
kit: `kit/README.md`, phase cards/templates reference canonical SPEC; archived spec non-authoritative.

§A — LeWM-first appendix

Reference-first adaptation, not claimed exact reproduction until upstream revision and preprocessing pinned P0. Raw game observations remain128 RGB; proposed network input224 RGB matches paper setting. Resize details and normalization audited before build; no invented pixel information. Small-object retention evaluated before changing patch size or adding dense tokens.

symbol|meaning
---|---
B|independent trajectory windows per optimizer batch
H|context length3 decisions initially
D|latent dimension192
oₜ|rendered frame before action aₜ
aₜ|one of9 actions applied for4 native frames
zₜ|projected CLS representation E(oₜ)
ẑₜ₊₁|F(z history, action history), predicted next representation
Z|embeddings with shape time×B×D for SIGReg
λ|SIGReg weight; released config0.09, paper default0.1; pin chosen reference profile

component|proposed reference configuration|output excluding B
---|---|---
Frame encoder|ViT-Tiny, patch14,12 layers,3 heads,width192; no pretrained weights|CLS192 per frame
Encoder projector|linear192→2048, BatchNorm, GELU, linear2048→192|zₜ192
Action embedder|one-hot9→learned projection→192; recorded categorical action, not continuous magnitude|192 per decision
Predictor|causal6-layer transformer; hidden192;16 heads with64 internal dimensions/head; MLP2048; dropout0.1; AdaLN-zero action conditioning|192 per context position
Predictor projector|linear192→2048, BatchNorm, GELU, linear2048→192|ẑₜ₊₁192
Training objective|L = mean squared prediction error + λ·SIGReg(Z)|scalar
SIGReg|1024 random projections;17 integration knots; timewise statistic across B|scalar
Diagnostic decoder|P5 train-only fitting to frozen features; not part of LeWM objective|rendered RGB probe
Controller|P7 chooses branch-specific interfaces after world-model acceptance|one of9 actions

Unicode prediction objective: L_prediction = mean(‖ẑₜ₊₁ − zₜ₊₁‖²). Both sides differentiable during world-model training. Future frames only targets; predictor causal. Teacher forcing in training distinguished from recursive open-loop evaluation. Zero detached targets is intentional, not a bug.

Core objective has exactly2 terms. No inverse/action-classification, pixel reconstruction, masked-DINO, EMA or value/reward losses. Reconstruction probes train AFTER freezing world model; diagnostic failures prompt reviewed ablations, not automatic extra loss terms.

Reference optimizer proposal: AdamW lr5e-5, decay1e-3, clip1; scheduler pinned from reference P0; batch128 reference versus practical fit measured P3. Gradient accumulation does not reproduce SIGReg batch statistics or BatchNorm behavior; changing batch size is an explicit adaptation. Report actual batch and per-time sample count. SIGReg receives post-projector features; do not L2-normalize them or silently apply prior LayerNorm contract.

BatchNorm in reference projectors requires explicit causality audit: training joint-frame normalization can mix sample/time statistics; acting/evaluation use frozen running statistics. P3 verifies future substitution cannot affect current outputs in evaluation and records train-mode coupling. No claim of full train-mode causal isolation without evidence.

Small-data/low-diversity failure remains possible even without constant collapse. Check statistical diversity and recoverable rendered detail separately. Predictability alone can favor background or slow features; no guaranteed object separation or collision awareness.

§P — canonical phase map
id|status|mode|goal|depends_on|gate
---|---|---|---|---|---
P0|ready-for-review|design|Review LeWM adaptation and pin reference|—|No input-authority ambiguity; architecture/preprocess/splits/budget draft reviewed.
P1|unopened|implementation|Implement pixel/action sequence boundary|P0|Whitelist, cadence, reset/truncation, split and artifact tests pass.
P2|unopened|collection|Collect and freeze dynamics corpus|P1|Hashed corpus and rendered scene coverage accepted; no learner updates.
P3|unopened|implementation|Implement LeWM core and diagnostics|P2|Reference numerical/gradient parity, no-EMA path, checkpoint and resource smoke pass; P4 gates locked.
P4|unopened|training|Train world model independently of controllers|P3|Valid noncollapse/dynamics screen; accepted checkpoint frozen for independent probes; no policy claim.
P5|unopened|implementation|Implement frozen-model probes and rollout evaluation|P4|Train-only decoder/readouts, common-space baselines, no gradient leakage or actual-future rollout inputs.
P6|unopened|diagnostic fitting/evaluation|Validate retained detail and open-loop dynamics|P5|Detail/temporal/prediction evidence accepted; limitations explicit; failed core blocks controllers.
P7|unopened|deferred design|Scope planning, policy and hybrid comparisons|P6|Survival objective, discrete search, DDQN/PPO fairness and controller protocols resolved; separate branch phase specs.
P8|unopened|deferred implementation|Implement one accepted controller branch|P7|Selected branch tests/smoke pass; training remains separate.
P9|unopened|deferred training/evaluation|Screen one accepted controller branch|P8|Frozen protocol comparison complete; no cross-branch coding or holdout selection.
P10|unopened|deferred training/evaluation|Confirm matched controller choices|P9|At least5 fresh seeds; one sealed holdout opening; paired uncertainty and timing.

P7-P10 are placeholders, not implementation-ready controller specifications. Each controller branch requires its own coding→training→evaluation subphases before P8 opens. They do not block P0-P6. Planning, DDQN, PPO and hybrid comparisons all remain intended downstream work; PPO is the confirmed alternative to DDQN; no unnamed controller slot. No controller is mandatory to validate LeWM.

§M — MVP delivery sequence (current scope)

MVP proves engineering path and observability, not learned Dodge understanding. User instruction overrides skill defaults against agents/dashboard; Luna max agents own bounded modules, parent validates all code. No training/coding blend: M1 completes and freezes before M2-M4; source defect stops run and reopens M1 with new identity. Code scaffolding may include diagnostic decoder before research P4 because M1 validates interfaces only; no scientific P6 acceptance implied.

id|status|mode|goal|depends_on|gate
---|---|---|---|---|---
M0|accepted|references/design|Pin upstream clone and paper Markdown; crosswalk paper/code; freeze MVP interface|—|Sources/hashes/license recorded; adaptation profile explicit
M1|accepted|implementation|Model/data/trainer/probes/dashboard plus source parity tests|M0|Parent code review, targeted/full tests, lint and narrow smoke pass
M2|accepted|collection|Small native train/validation corpus|M1|<=256 native decisions total; disjoint seeds; valid window hashes
M3|accepted|bounded training|Exercise world-model training/checkpoint/metrics|M2|<=32 optimizer steps; Colab T4 reference architecture; engineering-only; no quality pass
M4|accepted|diagnostic fitting/evaluation|Frozen decoder and direct feature/attention/future views|M3|<=32 decoder updates; model hash stable; validation frame/future alignment and labels verified
M5|accepted|serve/review|Read-only no-scroll dashboard; local+Tailscale access|M4|1280×720/1366×768 no-scroll check, endpoints safe, real artifacts shown

I.mvp: namespace CLI collect/pretrain/probe/dashboard; no controller. Dashboard surfaces pred/SIGReg loss, feature spread/rank, gradient norm, throughput, step/phase/provenance and explicit diagnostic qualification. Views: actual current/future pixels, decoded current/predicted future, encoder CLS attention and latent feature plot. Random/tiny/undertrained outputs labeled; no visualization of nonexistent object masks.


§E — execution envelopes

MVP amendment authorizes bounded M0-M5 sequence below; original scientific phases retain their acceptance gates. Coding checks <=32 optimizer updates/invocation, <=256 native decisions, at most3 timing repetitions; no campaign. P2 initial corpus cap250000 train transitions; validation collection separately bounded in P0. P4 initial screen2000 updates/condition across3 seeds; extension <=20000 updates only separate accepted screen decision. P5 fitting smokes obey coding caps. P6 diagnostic fitting budget locked P5 before fitting; may train probes only, never LeWM. P7 defines controller budgets before branch execution; previous DDQN campaign settings no longer active.

Device/time/memory/storage caps, actual SIGReg batch, data split sizes and stopping rules locked before run. Local lightweight checks only; Colab T4 reference model and decoder runs. Actual CUDA device recorded; no silent CPU fallback. Float32 feasibility smoke explicit deviation from upstream bfloat16; actual batch recorded, no accumulation equivalence claim. Frozen code/data/protocol in every learning phase; defect stops affected execution and reopens owner P1/P3/P5. Metrics from invalid revisions retained, never patched into same run identity.

§V
V1|Learner whitelist excludes arbitrary native info. No hidden-state training or evaluation labels; rendered-pixel visual review described as qualitative evidence.
V2|Entire episodes split before windows. Same-lane/cadence history/futures, no reset crossing or lookahead. Terminal versus truncation semantics explicit; bootstrap only according to accepted native contract.
V3|Train/representation-validation/inner-policy/final-holdout game seeds disjoint and immutable; native seeds in0..32767, no clamping. Holdout absent from all pretraining/selection/teacher updates. Learner seeds independently declared.
V4|LeWM has no certified player-attention map. Evaluate body/trail, motion and hazard retention through frozen probes and rendered clips; saliency is diagnostic only.
V5|LeWM screen includes prediction+SIGReg, random/constant controls and bounded prediction-only collapse diagnostic. Any alternate encoder/objective separate reviewed ablation; no multitask bundle.
V6|Track feature variance/effective rank, SIGReg, prediction/persistence errors, per-module gradients and actual update/weight ratios; P3 locks floors before P4. Noncollapse alone never acceptance.
V7|Detail evaluation covers body/trail, filled/hollow squares, adjacent objects, pattern boundaries and halos on fixed rendered validation clips. Pixel error stratified using pixels only; compare reconstruction-only and constant-image baselines. No semantic segmentation score without labels.
V8|P6 prediction compares same frozen model target space against persistence and action/temporal-shuffle controls. Never use actual future embeddings in open-loop predictor context. Different models compared with common frozen evaluation readout or explicitly separated metrics.
V9|No exploration bonus, new game semantics, reward changes, masking or crop/color/flip augmentation in initial LeWM candidate.
V10|No controller during P0-P6. Initial later branches share selected frozen world model; DDQN target network is separate, PPO rollout distribution separate; no EMA teacher in LeWM.
V11|P7 establishes fair planning/DDQN/PPO/hybrid comparisons: equal transition exposure including corpus, separate pretraining/controller compute and decision latency. PPO uses on-policy data for its policy updates; no pretending offline replay updates are on-policy.
V12|Final comparison uses >=5 fresh matched learner seeds beyond screen seeds. Freeze difficulty/patterns/reward/cadence/evaluation cap and checkpoint selection rule. Greedy neutral/fixed/random controls, per-seed survival/action counts, Q-gap scene response and censoring reported.
V13|Positive policy claim requires positive95% paired CI on mean survival difference versus strongest predeclared baseline, plus scene-sensitive greedy behavior; uncertainty method accounts for learner seeds, not frames as independent samples. Otherwise inconclusive/negative.
V14|Training code/data/protocol hashes frozen. Source change stops affected run, reopens owning coding phase, preserves evidence and requires new run identity. Diagnostic fitting cannot mutate LeWM.
V15|Checkpoint restores world-model/projector/BatchNorm state, all active optimizer/scheduler/RNG including SIGReg projections, data position and counters. Evaluation uses fixed model statistics and independent RNG.
V16|New variant imports/startup never change legacy defaults, entrypoints, checkpoints or artifact roots. No namespace promotion via editing legacy SPEC. New architecture/ABI rejects incompatible state before tensor mutation.
V17|Phase status distinguishes unopened/active/implementation-complete/evidence-ready/accepted/rejected/inconclusive. Accepted phase != positive scientific result. Only explicit acceptance or already-authorized auto-advance opens successor; failed prerequisite never bypassed.
V18|Implementation closures run meaningful narrow tests plus `scripts/uv-run pytest`, `scripts/uv-run ruff check .`, unique32-step legacy Colab T4 smoke and applicable new-variant capped smoke. Collection/training closures verify hashes and evaluate evidence, not write implementation.
V19|Root SPEC links scoped authority. New variant SPEC owns all phase/task/bug IDs; kit links only. No fake acceptance reports, fabricated metrics, runnable-looking unimplemented CLI commands or completed task marks.
V20|P2 has no optimizer; P4 only LeWM optimization; P6 only frozen-model diagnostic fitting/evaluation. P8 coding not controller campaign; P9/P10 later authorized controller work only.
V21|Missing rendered scene coverage blocks P6 capability claims; incomplete corpus revisits P2, collector defects P1. Easy/no-pattern survival never final evidence.
V22|Failed P6 core quality blocks controller phases. A validated world model does not force planner selection; downstream branches may yield negative/inconclusive results without altering shared core.
V23|Final holdout opens once after controller recipes/checkpoints frozen; P4/P6 use representation-validation only. No new tuning after holdout; fresh experiment needs untouched holdout.
V24|Servers, GPU allocation and external jobs require explicit phase execution scope; protect all existing processes/worktrees. User-requested scaffolding never implies long-run authorization.
V25|Proof of phase gate includes commands, return codes, device/runtime/source hashes, artifact paths, limitations, visual review and acceptance authority. Failures considered for §B backprop; do not weaken gates to advance.
V26|LeWM loss exactly prediction+SIGReg; target embeddings remain attached in training. No EMA, stop-gradient, inverse loss or reconstruction gradients. Frozen diagnostic decoder/readout cannot update world model.
V27|Reference BatchNorm/projector/SIGReg axes preserved and explicitly tested; batch accumulation or feature normalization changes not treated as equivalent. Pin upstream commit/config and enumerate Dodge adaptations.

V28|Dashboard fitting no document scroll at1280×720/1366×768; tabs preserve access; bounded polling and read-only routes; path traversal/symlink escape rejected; malformed/incomplete run remains visible without invented metrics.
V29|MVP visualization exposes actual model tensors/decoded results tied to exact checkpoint and observation; decoder fits only training pixels with frozen world model and its BatchNorm buffers; projected-z decoder consumes projected-z including predictedz.
V30|Source reuse retains MIT notice and pinned SHA; paper Markdown retains attribution/license and points to exactv3 HTML. Source crosswalk identifies changes, defaults and missing empirical replication.

V31|Autoregressive rollout with H observed frames and A actions returns A+1 latent frames; final action contributes final prediction. Action-trace test guards upstream jepa.py final-state step.

V32|Colab dependency installation precedes fresh-process model imports; worker reruns model/native checks on actual T4. Completion requires worker sentinel plus retrieved artifacts, not Colab CLI exit status alone.

I.scenario|Strict version1 TOML → resolved ScenarioConfig → native constructor rules; collector --scenario PATH; complete resolved config+SHA256 retained in dataset manifest and run provenance. Presets empty/permanent-patterns/normal-easy/standard. Pixel/action learner input unchanged.
V33|Four image panes remain; full square bitmap contained inside frame body at1280×720 and1366×768; no intrinsic-size clipping.
V34|Default scenario retains existing native trajectories and canonical wire8 bytes. Nondefault rules serialized in explicit wire9 extension, included in state hash, preserved through reset/restart/restore.
V35|Enemy-none blocks scheduled and anti-idle spawns. Normal-only blocks special personalities. Permanent pattern initial rectangles fully shown, stationary and persistent. Invulnerable suppresses player collision/death events; ordinary mode remains lethal.
V36|Scenario TOML rejects unknown keys/versions/types/modes and invalid pattern IDs. Resolved config/hash propagated into corpus and run; scenario metadata never encoder input. Independent seeds still disjoint.

§S — scenario delivery
S0|complete|dashboard sizing|Keep four-panel layout, show full aspect ratio|Browser geometry + actual screenshot
S1|complete|implementation|Native rules/snapshot extension, TOML presets/loader and collector plumbing|Native invariants + baseline parity + Python schema tests
S2|complete|verification|Bounded configured collection and artifact audit|Scenario config/hash preserved; no world-model training required

§G.script — requested extension, design pending
G.script1|Author bounded practice trajectories and capture reached goals as native-rendered frames/clips. Script coordinates and progress remain outside learner inputs. Actor scope ? player/enemies/both; [contract and proposed gates](kit/SCRIPTED-PRACTICE.md).
G.script2|Preserve terminal transition; no death-to-reset windows. Later controller termination handling separate from LeWM representation objective. Moving-goal objective requires explicit Dodge adaptation of paper §3.2.
G.script3|G0 contract → G1 implementation → G2 bounded collection → G3 later controller design. No script implementation or additional training opened by this note.

§T
id|status|task|cites
---|---|---|---
T1|.|P0: Pin upstream revision, audit objective/config and decide pixel resize/action encoding.|C8,C10,V14,V17,V25,I.kit
T2|.|P0: Lock corpus splits, resource/accounting rules and diagnostic questions; reserve unknown controller choice.|C8,C10,V14,V17,V25,I.kit
T3|.|P0: Publish LeWM architecture review; hand off data implementation only.|C8,C10,V14,V17,V25,I.kit
T4|.|P1: Implement raw-pixel/action whitelist and episode-contained sequence loader.|C8,C10,V14,V17,V25,I.kit
T5|.|P1: Test exact cadence, truncation/reset, split isolation and artifact namespaces.|C8,C10,V14,V17,V25,I.kit
T6|.|P1: Freeze collector revision/protocol after bounded verification.|C8,C10,V14,V17,V25,I.kit
T7|.|P2: Collect accepted pixel/action corpus with no optimizer.|C8,C10,V14,V17,V25,I.kit
T8|.|P2: Audit rendered detail/halo/motion coverage and sequence hashes.|C8,C10,V14,V17,V25,I.kit
T9|.|P2: Freeze dataset/splits and publish acceptance evidence.|C8,C10,V14,V17,V25,I.kit
T10|.|P3: Implement encoder, projector, action-conditioned causal predictor and SIGReg.|C8,C10,V14,V17,V25,I.kit
T11|.|P3: Test attached-target gradients, statistic axes, BatchNorm/evaluation causality and checkpoint equivalence.|C8,C10,V14,V17,V25,I.kit
T12|.|P3: Calibrate diagnostics and resources; freeze world-model code and P4 protocol.|C8,C10,V14,V17,V25,I.kit
T13|.|P4: Run only frozen LeWM screen and declared controls.|C8,C10,V14,V17,V25,I.kit
T14|.|P4: Measure noncollapse, prediction and sample diversity on representation-validation.|C8,C10,V14,V17,V25,I.kit
T15|.|P4: Freeze accepted world model; publish limitations without survival claims.|C8,C10,V14,V17,V25,I.kit
T16|.|P5: Implement detached decoder/readouts and open-loop rollout evaluator.|C8,C10,V14,V17,V25,I.kit
T17|.|P5: Test probe gradient isolation and persistence/action/temporal controls.|C8,C10,V14,V17,V25,I.kit
T18|.|P5: Freeze probe fitting budget and visual review protocol for P6.|C8,C10,V14,V17,V25,I.kit
T19|.|P6: Fit diagnostic probes on TRAIN only with world model frozen.|C8,C10,V14,V17,V25,I.kit
T20|.|P6: Evaluate unseen detail, tails, hollow enemies, boundaries, halos and temporal futures.|C8,C10,V14,V17,V25,I.kit
T21|.|P6: Publish core acceptance or failure; stop before controller work.|C8,C10,V14,V17,V25,I.kit
T22|.|P7: Define survival-based planning cost from ordinary outcomes; learned-policy alternatives are DDQN and PPO.|C8,C10,V14,V17,V25,I.kit
T23|.|P7: Specify separate planning, DDQN, PPO and hybrid experiments with fair data/compute accounting.|C8,C10,V14,V17,V25,I.kit
T24|.|P7: Amend individual controller coding/training phase contracts before opening P8.|C8,C10,V14,V17,V25,I.kit
T25|.|P8: Implement only one controller branch selected by accepted P7 protocol.|C8,C10,V14,V17,V25,I.kit
T26|.|P8: Verify action legality, frozen core and branch-specific objective/data semantics.|C8,C10,V14,V17,V25,I.kit
T27|.|P8: Run bounded smoke and hand off fixed controller training/evaluation protocol.|C8,C10,V14,V17,V25,I.kit
T28|.|P9: Run one accepted controller experiment without source mutation.|C8,C10,V14,V17,V25,I.kit
T29|.|P9: Evaluate inner behavior, survival and latency versus declared controls.|C8,C10,V14,V17,V25,I.kit
T30|.|P9: Freeze confirmation recipes and unused learner seeds without opening holdout.|C8,C10,V14,V17,V25,I.kit
T31|.|P10: Run matched frozen controller recipes on at least5 new learner seeds.|C8,C10,V14,V17,V25,I.kit
T32|.|P10: Freeze checkpoint hashes and evaluate final holdout once.|C8,C10,V14,V17,V25,I.kit
T33|.|P10: Publish complete paired results and limitations; no automatic extension.|C8,C10,V14,V17,V25,I.kit

T34|✓|M0: pin references and architecture/code crosswalk|V30
T35|✓|M1: parent-validated Luna model/data/dashboard code plus trainer integration|V26-V30
T36|✓|M2: bounded native corpus with disjoint splits|V1-V3
T37|✓|M3: bounded world-model train/checkpoint/metrics|V14,V15,V26
T38|✓|M4: frozen diagnostic fitting and source-linked model outputs|V4,V8,V29
T39|✓|M5: serve/review no-scroll dashboard over Tailscale|V28,V29

T40|✓|S0: square-image containment, retain four-panel layout|V33
T41|✓|S1: native scenario controls and snapshot compatibility|V34,V35
T42|✓|S1: strict TOML presets and collector/run provenance|V36
T43|✓|S2: configured collection and regression verification|V34-V36

§B
id|date|cause|fix
---|---|---|---

B1|2026-09-14|Dependency sync invoked through scripts/uv-run, which runs commands rather than uv subcommands; OS sync rejected --extra|Use direct uv sync; invocation mistake, no new model invariant

B2|2026-09-14|CUDA runner edit introduced extra closing bracket; py_compile caught before execution|Bracket corrected; mechanical typo, existing V18 sufficient
B3|2026-09-14|Initial runner formatting and generated remote-script import placement failed Ruff|Format files; remote phase body moved into main; existing V18 sufficient
B4|2026-09-14|New test_dashboard basename collided legacy test module during full-suite collection|Rename test_lewm_dashboard; existing V18 catches
B5|2026-09-14|Torch zip writer rejects extensionless dot-prefixed checkpoint path|Write through file handle, flush/fsync then replace; resume/probe integration tests guard existing V14/V26
B6|2026-09-14|Rollout helper omitted upstream final-state prediction; action-trace test failed on A-H instead of A-H+1 outputs|V31; loop includes final action; v1 Colab stopped during setup, new identity required
B7|2026-09-14|Colab notebook retained older NumPy after pip upgrade; cold-process checks passed but notebook ViT import failed; CLI returned0 despite cell exception|V32; isolated worker after installation, explicit completion marker; v2 data/logs preserved, no optimizer steps
B8|2026-09-14|Real-data UI rounded tiny nonzero spread to0 and CSS display overrode hidden empty-state overlays|Scientific notation; explicit hidden rule; real-artifact browser assertions and both viewport checks pass; V28 sufficient
B9|2026-09-14|Browser retry started before restarted listener bound port|Health check before browser navigation; orchestration race, no new model invariant
B10|2026-09-14|Whitespace check flags trailing spaces in exact downloaded paper HTML|Preserve source bytes/hash; authored-file whitespace check excludes reference HTML; no new invariant
B11|2026-09-14|Grid intrinsic image sizing enlarged square content beyond frame body and clipped it into horizontal shape|V33; bounded rows and absolutely contained images, browser square-content bounds
B12|2026-09-14|Rust constructor test omitted new scenario defaults; workspace compile caught missing arguments|Supply explicit defaults in direct Rust call; mechanical migration, V34 regression suite sufficient
B13|2026-09-14|Scenario launcher additions exceeded Ruff line length|Format launcher; mechanical formatting, V18 sufficient
B14|2026-09-14|Parent review found permissive direct dataclass types and scenario claims on injected environments|Strict construction and reject conflicting injection; V36 sufficient, schema tests cover types
