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

§G.script — scripted practice delivery
G.script1|Author bounded practice trajectories and capture reached goals as native-rendered frames/clips. Script coordinates and progress remain outside learner inputs. Actor scope: both player and enemies; [contract and proposed gates](kit/SCRIPTED-PRACTICE.md).
G.script2|Preserve terminal transition; no death-to-reset windows. Later controller termination handling separate from LeWM representation objective. Moving-goal objective requires explicit Dodge adaptation of paper §3.2.
G.script3|G0 contract → G1 implementation → G2 bounded collection → G3 later controller design. User selected both actors; G0 reviewed, G1/G2 complete; G3 controller design deferred. No model training.

V37|Practice player moves execute ordinary nine-action/four-frame native physics; initial placement precedes first observation; scripts record actual actions. MoveTo timeout never successful goal.
V38|Scripted square enemies remain in native game/render/collision state; static or bounded linear paths. Driver config/coordinates never learner inputs. Externally driven native mode3 suppresses ordinary spawns/AI; full practice replay requires script+seed, not a standalone game snapshot.
V39|Practice stops on death, script completion or bounded timeout; death takes precedence over a coincident waypoint timeout. Keep death outcome, never reset across a trajectory. Goal valid only on live successful script completion; captured motion clip ordered and source-hashed.
V40|Default native modes0-2 retain behavior/wire compatibility. Practice authoring <=256 decisions,4 frames/action,<=64 commands,<=32 enemies,<=64 path segments/actor. Validate native inputs independently of Python.
V41|Practice generation separate from corpus ingestion/model training. Immortal collision outcomes and lethal collision outcomes have different dynamics; configuration provenance retained, no automatic mixed training.

§D — practice corpus bridge
D1|accepted|implementation|Import explicit train/validation practice recordings into existing bounded corpus; no generation or optimizer|G2|Parent review; exact NPZ bytes, hashes, split/terminal/provenance tests; full Python regression
D2|accepted|collection verification|Freeze D1; generate one validation recording and import with existing train recording|D1|<=256 new native decisions, both splits load; source/array hashes agree; no model updates
D3a|accepted|implementation|Explicit practice overfit envelope, frozen-corpus T4 launch, one-step controls|D2|Focused/full Python tests; source parity unchanged; no run until code frozen
D3|accepted|training|Practice-only overfit diagnostic|D3a|Fresh reference LeWM,512 updates,batch8,seed42,float32,T4; frozen21-transition D2 corpus; no corpus growth
D4|accepted|diagnostic fitting/evaluation|Frozen decoder and prediction controls|D3|256 decoder updates,batch8; all train/validation windows; checkpoint stable; no scientific P4/P6 acceptance
I.practice_dataset|CLI `python -m dodge_native_game.variants.pixel_repr_ddqn.practice_dataset --output PATH --train PATH [PATH...] --validation PATH [PATH...]`; explicit existing captures; new output only.
V42|Practice import preserves episode bytes and original manifest hashes; validates source schema/config/cadence/action trace/count/end flags and NPZ payload. Reject path escapes, incomplete/tampered inputs, duplicate seeds or identical episode bytes across splits, mixed invulnerability and >256 transitions. Both splits need history3 windows; short episodes retained without fabricated windows.
V43|Practice provenance and goals remain metadata, never learner inputs. Import performs no native steps or optimization. Publish validated corpus atomically; failure leaves no destination. Scripted enemy future instructions unobserved by learner; no claim that arbitrary unseen path turns are predictable.

V44|Extended budget opt-in `practice-overfit-v1` only: fresh reference CUDA,512 model updates,batch8,seed42, no resume; practice corpus required. Default MVP caps unchanged. Decoder extension requires matching checkpoint experiment;256 updates,batch8,CUDA. Frozen source+dataset manifest+protocol hashes; no silent collection when supplied corpus.
V45|One-step dynamics use observed context only; compare predicted next to same-model encoded next, current-latent persistence, wrong-action `(a+1)%9` control. Report action sensitivity; zero persistence denominator → null ratio. No automatic quality pass; wrong-action control not a causal counterfactual rollout. Read-only evaluation leaves model/RNG unchanged.

§N — frozen normalization audit
N1|accepted|implementation|Factorial BN/dropout/position audit and train-only disposable buffer interventions|D4|Mode/RNG/buffer restoration, no validation calibration, source review, full Python/Ruff
N2|accepted|diagnostic evaluation|Execute frozen checkpoint audit on T4|N1|Zero optimizer/native/decoder updates; source/data/checkpoint hashes; matched sampler indices; original model/checkpoint unchanged
V46|Audit uses identical cached deterministic encoder CLS inputs; encoder BN/predictor BN/dropout switched independently, per-position errors reported. Three dropout seeds2026-2028; original/wrong actions share masks. Train modes diagnostic only, never deployment evidence; target spaces can change with encoder BN, compare ratios within condition.
V47|Calibration changes buffers only on disposable copies; exact training-window population moments, no validation fitting. Predictor-only intervention preserves encoder/target space. Original buffers/modes/RNG restored on exceptions; no optimizer, no checkpoint write, no promotion of calibrated copies. Future substitution changes only final-frame CLS; evaluation context invariance must hold.

§K — calibrated diverse-practice screen
K1|accepted|implementation|Inference-only encoder calibration export; bounded diverse suite; paired raw/calibrated runner|N2|Train-only/weight-preservation/resume rejection/causality tests; suite determinism and coverage; full Python/Ruff
K2|accepted|collection|Freeze16 explicit practice captures|K1|12train seeds500-511,4validation1500-1503;16decisions each;256total; source/config/episode hashes, rendered review
K3|accepted|training|Fresh diverse-practice-v1 screen|K2|Reference T4,float32,512updates,batch8,seed42; unchanged objective/optimizer; no extension
K4|accepted|calibration and diagnostic fitting/evaluation|Paired original/calibrated checkpoints and decoders|K3|Encoder population moments from train only;256 decoder updates per condition; persistence/wrong-action metrics; no controller promotion
V48|Calibration export preserves original checkpoint and learned weights; only encoder BN running_mean/running_var change. Derived checkpoint inference_only, no optimizer/RNG state; trainer rejects resume. Parent/data/derived hashes recorded; original and calibrated runs separate; original decoder never reused for changed latents.
V49|Diverse suite stays native/pixel/action-only:12train+4validation captures,16decisions each,all9training actions, varied starts/static/moving enemies; all invulnerable, difficulty1, no patterns. Explicit episode splits/hashes, total256 cap. New dataset comparison with prior21-transition corpus not a matched generalization claim.
V50|practice-diverse-v1 opt-in fresh reference CUDA512updates,batch8,seed42, no resume; default MVP unchanged. Calibration/evaluation after training; no validation calibration or selecting raw/calibrated checkpoints by validation. Report both and limits; no automatic long run.

§L — batch-size screen and pixel controls
L1|accepted|implementation|Batch32 protocol, retained128-step checkpoint, pixel diagnostics and paired runner|K4|Envelope/retention tests; pixel controls analytic fixtures; full Python/Ruff
L2|accepted|training|Fresh reference batch32,512updates,seed42 on same frozen corpus|L1|One T4,float32,same objective/optimizer; retain128 and512; no new collection
L3|accepted|calibration and diagnostic fitting/evaluation|Raw/calibrated128 and512 checkpoints; same256-step decoder each; old batch8 calibrated decoder read-only baseline|L2|All validation windows/persistence/wrong actions; train-mean image and changing-pixel controls; per-episode reporting; square dashboard
V51|practice-batch32-v1 exact fresh referenceCUDA512updates,batch32,seed42, imported corpus;128-step retained optimizer snapshot immutable.128×32 equals prior512×8 sampled windows;512-step comparison matches updates but sees4× windows. Neither isolates BN causally; reference architecture/SIGReg/optimizer unchanged.
V52|Pixel diagnostics only:32×32 area targets, train-window current-frame mean baseline; validation-only abs(next-current) meanRGB>1/255 mask. No mask/mean input or loss to LeWM/decoder. Report global/changed-region SSE per channel-pixel, counts/null empty masks, per-episode metrics. Frozen model/decoder/modes/RNG preserved; static HUD recognition alone not entity retention evidence.
V53|L resource cap:one T4,3000s worker,512 modelupdates,4×256 decoderupdates; baseline checkpoint/decoder read-only,no refit. Fixed old corpusSHA7769398938b2be30934779fc5851e767653df7b31b2e7fad7a2a0adcd80e6a1f. Stop source defect/OOM/nonfinite; no automatic batch fallback/extension/controller promotion.

§O — full-resolution frozen decoder study
O1|accepted|implementation|Native128 query decoder; frozen train cache; resumable optimizer/sampler; resolution-aware diagnostics|L3|Shape/patch order/gradient/cache/resume/native-metric tests; full Python/Ruff
O2|complete|diagnostic fitting|One query decoder256updates; frozen batch32/512 calibrated checkpoint|O1|One T4,3000s worker; seed904,batch8windows; no LeWM/collection updates
O3|complete-negative|evaluation|Native128 primary,32 secondary; all56 validation windows and8 fixed views|O2|Verify hashes/tensors; trainmean/persistence/changing pixels; square dashboard. Promising only if changing-region current reconstruction beats trainmean and views recover scene-dependent detail beyond HUD. Then allow same decoder continuation to512 total; otherwise stop256. Validation exploratory, no generalization claim.
V54|User requests1:1 reconstruction: query decoder target/render native128RGB; prior32 MVP shortcut. Train loss plain native128MSE;32 metrics historical only. Existing historical32runs/defaults unchanged.
V55|Fixed finalbatch32/512 encoder-calibrated LeWM; projected192latents cached train only. One fresh query decoder hidden128,3blocks,4heads,64queries,patch16,no dropout.256updates,batch8windows×4frames,AdamW1e-3/decay0.01,seeds904init/903sampling(discard2). Optional continuation to512 total only after O3 promising review; resume weights/optimizer/sampler exactly. No validation gradients, new game/transitions/world updates. Earlier two-decoder2048 plan superseded by user.
V56|Paper decoder absent pinned repository; structure AppendixD, reuse releasedFeedForward. Native128/64queries/projectedlatent input local adaptations; paper224/196queries/preprojectionCLS. No exact reproduction claim. Preserve original world tensors/BN/files; detached train cache; decoder-only gradients. Source/data/resume hashes recorded.


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

T44|✓|G1: native player/action and enemy/path practice driver|V37-V40
T45|✓|G1: strict practice config, replay/goal artifact generator|V37-V41
T46|✓|G2: bounded replay, goal and terminal verification|V37-V41

T47|✓|D1: practice corpus importer and boundary/provenance checks|V1-V3,V39,V41-V43
T48|✓|D2: frozen importer and real practice corpus verification|V14,V42,V43

T49|✓|D3a: practice overfit envelope, frozen corpus launch and dynamics checks|V14,V20,V44,V45
T50|✓|D3/D4: bounded T4 overfit and frozen evaluation|V14,V20,V44,V45

T51|✓|N1: implement frozen normalization and causality audit|V14,V46,V47
T52|✓|N2: run and interpret T4 audit; preserve failed-run evidence|V14,V46,V47

T53|x|K1: calibration export, diverse suite and paired evaluation integration|V14,V48-V50
T54|x|K2: collect/hash/review bounded diverse practice corpus|V49
T55|x|K3/K4: T4 screen and original/calibrated diagnostics|V14,V48-V50

T56|x|L1: bounded batch32/checkpoint retention and pixel controls|V48,V51-V53
T57|x|L2/L3: matched-exposure/update T4 screen and visual review|V48,V51-V53

T58|x|O1: native-resolution frozen decoder implementation|V48,V52,V54-V56
T59|x|O2/O3: T4 decoder fits and held-out nativepixel review|V48,V52,V54-V56

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
B15|2026-09-14|Local stopping-position driver could not settle fixture96,80 at four-frame cadence; native trace stopped near92,78|Use reachable fixture92,78; preserve2px position and0.2 speed tolerance and explicit timeouts, V37 sufficient
B16|2026-09-14|Parent review found coincident death and waypoint budget expiry could report both terminal and script failure|Death takes precedence; native regression uses fatal final-budget MoveTo, V39 sufficient
B17|2026-09-14|Generator review found live script completion lacked truncation and direct action dataclass accepted booleans|Separate game termination from collection stop; reject boolean action indices, V39/V40 sufficient
B18|2026-09-14|Initial practice test import formatting failed Ruff|Organize imports and format authored files; V18 sufficient

B19|2026-09-14|Importer/tests first Ruff check found import ordering and long lines|Format authored files; existing V18 sufficient; subsequent Ruff clean

B20|2026-09-14|Practice envelope first Ruff check found long error message and worker import ordering|Wrap literal and organize imports; existing V18 sufficient

B21|2026-09-14|Dynamics test expected extra latent dimension; evaluator lacked explicit RNG isolation|Correct shape; fork CPU/device RNG, reject nonfinite metrics; V45 sufficient

D.result|D3/D4 execution complete; negative dynamics evidence: prediction/persistence MSE ratio16.1043 train,10.8561 validation. Wrong actions slightly better. No controller opening or training extension; matched train/eval normalization diagnostic next.

B22|2026-09-14|Audit lint found unbound loop closures and formatting issues|Bind candidate/latent defaults explicitly; format literals/contexts; existing V18/V47 sufficient

B23|2026-09-14|Launcher review found package import eagerly requires native module even for offline diagnostics|Include/build native dependency before remote tests; no native game steps; existing V18 sufficient

B24|2026-09-14|Colab191MiB archive upload failed twice with TLS EOF before any diagnostic execution;8MiB chunk succeeded|Chunk host transfer, hash reassembled frozen archive; diagnostic code/checkpoint unchanged, same run identity valid

N.result|Encoder running-stat intervention restores train fit: ratio16.1043→0.01628; validation10.8561→1.14132 still fails persistence. Predictor-only intervention worsens; dropout/position not dominant. Train-mode future coupling confirmed; fixed-stat eval invariant. Original checkpoint unchanged; normalization export protocol next, no controller promotion.

B25|2026-09-14|Calibration first lint found long docstring, loop closure and unused test import|Bind pixel batch in capture closure, format and remove unused import; V18 sufficient

B26|2026-09-15|Browser review traversed image handles while polling replaced DOM; detached parent exception|Read all image bounds in one evaluate_all call; both viewports/tabs pass. Harness-only race, existing layout V sufficient; no product or frozen source change

K.result|Completed T4 screen512updates/194.79s;91 remote tests,310 local tests,legacy32-step smoke. Original train/validation prediction:persistence185.721/28.140; encoder-calibrated1.25434/0.996191. Calibrated wrong-action error+14.45%train/+4.73%validation. Validation persistence margin0.38%, weak one-seed evidence; decoded entities indistinct. Only two encoder buffers changed, all learned tensors preserved; inference-only export and separate decoders verified; T4 released. Engineering delivery accepted, scientific P4/P6 and controllers remain deferred.

R.batch|Pinned upstream train/lewm.yaml batch128 and paper AppendixD agree; batch32 intermediate T4 screen, not exact reproduction|https://arxiv.org/html/2603.19312v3#A4

B27|2026-09-15|Batch-screen first lint:long comparison string/import grouping|Shorten label and format imports; existing lint invariant sufficient

B28|2026-09-15|Probe integration hung: legacy fake dataset __getitem__ accepted unlimited indices, violating Python sequence iteration boundary|Fixture raises IndexError at len; real dataset already bounded. Existing episode-boundary invariant sufficient; stop duplicate task tests and rerun full suite

L.result|323local/104remote tests; T4 batch32/512 completed565.81s,peak6.591GB. Calibrated128train/val ratios0.79615/0.95307;512ratios0.20840/0.99491.512wrong-action penalty+249.30%train/+9.62%val. Changed-pixel prediction error improved7.45% vspriorcalibrated, all4episodes; still17.93%worse than pixelpersistence. Fixed train-mean beats global reconstruction; no clear entity outlines. Exact128×32 sampling prefix equals prior512×8; all learned tensors/calibration/decoder links verified. Artifacts retrieved,T4 released; no controller promotion.

B29|2026-09-15|Decoder-launcher first lint:long failure string/import grouping while study module pending|Split string;format imports aftermoduleexists; existinglint invariant sufficient

B30|2026-09-15|Study review:cache lacked explicit train-split guard and draft evaluated first decoder before second fit|Enforce traincache split; finishbothfits beforeevaluation; existingV55 covers; addcache rejectiontest

B31|2026-09-15|Bounded decoder launcher/current-region metrics exceeded Ruff line length|Format authored files; mechanical issue, existing V18 sufficient

O.validation|337 local tests pass; Ruff clean; exact split-fit resume equality; legacy32-step smoke expected bounded warnings. Source freeze before one T4 query256 fit.

B32|2026-09-15|Browser harness invoked project environment without Playwright|Use isolated uv --no-project --with playwright; no source mutation; existing browser layout check sufficient

O.result|One T4 query decoder256 updates completed;3.252s fitting,peak192290816B.337 local/118 remote tests; Ruff/smoke pass. Native128 validation current MSE0.00291185 vs trainmean0.00268604 (+8.41%); changed-current0.138592 vsmean0.137958 (+0.46%). Train current+10.34% vsmean; changed-current+2.28%. Eight views mostly background/HUD, no distinct player/enemies. Nextglobal error+24.74% vs persistence. Continuation gate failed: no512 fit. World SHA unchanged; optimizer256/sampler/source hashes verified;32 native PNGs, no-scroll browser at1280×720/1366×768. Artifacts retrieved,T4 released; source81251c5, archivea14e6d9a54a8890a1317472e0899fc90f9ca4c89bd1d4eb7ea3ad04a2bbd696a. Decoder versus representation bottleneck unresolved; no controller opening.


§O.continuation — user-directed decoder extension
O4|complete|diagnostic fitting|User sees emerging detail and explicitly requests continued iteration; resume existing query256 to512 total|O3|Same source/model/data/loss/batch;256 additional updates on one T4; saved optimizer/sampler; no world-model updates
O5|complete|evaluation|Compare identical eight views and all56 validation windows at256/512|O4|Native128 metrics and fixed mean/persistence controls; verify resume identity and unchanged world state; report measured gain separately from recognizable detail
O.authorization|User continuation supersedes earlier stop decision for this bounded extension. Original O3 negative result retained; no claim that original gate passed. Stop at512 total, evaluate, release T4.

O.continuation.result|Query256→512 same weights/AdamW/sampler;256 new updates,3.557s fitting,T4 peak192290816B.118 remote tests pass; implementation identical81251c5 (prior337 local tests). Validation current error−5.00%,changed-current−0.42%,next−4.57%,changed-next−0.53% vs256. Changed-current improves3/4 validation episodes. Current reconstruction still+2.98% vsfixedmean;changed-current+0.034%. Score clearer; distinct player/enemy tracking unverified. World/checkpoint unchanged; source/resume/optimizer512/sampler/pairedviews verified. Native128 dashboard passes both viewports; artifacts retrieved,T4 released. Source8f9ee8c,archivec9ea5037366c559e37c76d74fcf228e0bf6fbc9ff20fcebdc3e66a4d7be98618. No controller promotion; stop512.


§O.current — current-frame reconstruction at2048
O6|accepted|implementation|Extend bounded runner/launcher to2048 total from saved512 decoder|O5|Reject other resume step; exact optimizer/sampler continuity; native128 pixel-MSE unchanged; full Python/Ruff
O7|complete|diagnostic fitting|User requests2048 total;1536 additional query-decoder updates from512|O6|One T4,3000s worker; frozen world model/corpus/source; no collection or LeWM updates
O8|complete|evaluation|Primary focus current-frame reconstruction; fixed eight views and all56 validation windows vs512|O7|Train/validation current errors, mean controls, changed regions; decoded prediction retained as secondary existing diagnostic; verify resume/model/source hashes and release T4
O.current.authorization|Explicit user2048 request supersedes prior512 cap. Preserve old results; stop2048 total. Decoder fits observed pixels only; no prediction-loss tuning or controller work.

O.current.validation|338 local tests pass; Ruff and bounded32-step legacy smoke pass (expected bounded warnings). Parent-reviewed Luna max extension; no architecture/loss changes. Freeze before2048 continuation.

O.current.result|2048 complete:1536 new updates,18.347s fitting;338 local/119 remote tests. Train current MSE0.00126317 (−52.26% vs512),validation0.00340290 (+23.02%);validation changed-current0.138582 (+0.42%). Shapes emerge but positions often disagree with observed frames. Resume/optimizer2048/sampler/world/fixedviews verified; T4 released. Generalization remains weak; user explicitly requests8192 next.

§O.long — decoder fit to8192
O9|accepted|implementation|Permit8192 total only from saved2048 decoder|O8|Preserve existing resume envelopes; reject wrong prior step; full tests/Ruff
O10|complete|diagnostic fitting|User8192 request:6144 additional updates from2048|O9|One T4,3000s worker; identical pixel reconstruction loss/model/corpus; saved optimizer/sampler
O11|complete|evaluation|Current reconstruction focus; compare2048/8192 and retain512 reference|O10|Train/validation MSE and same eight images; separate visual detail from correct object placement; verify hashes/state and releaseT4
O.long.authorization|Explicit8192 request supersedes2048 cap. Stop8192 total; no world-model updates, collection, controller or prediction-specific tuning.

B33|2026-09-15|8192 launcher envelope exceeded line limit|Format launcher; mechanical issue, existing Ruff invariant sufficient

B34|2026-09-15|Standalone artifact image helper used system Python without Pillow|Use project environment; artifact-only invocation, existing verification sufficient

O.long.validation|338 local tests pass; Ruff clean; previous bounded smoke same task passed. Parent-reviewed Luna max resume extension; source freeze before8192 continuation.

O.long.result|8192 complete;6144 additional updates from2048,70.982s fitting,T4 peak192290816B.338 local/119 remote tests; Ruff clean. Train current MSE0.000763095 (−39.59% vs2048);changed-current−44.51%. Validation current0.00366236 (+7.62%);changed-current0.139783 (+0.87%). Sharper player-like tails/squares in some views, frequently misplaced. Current-frame generalization weak; no prediction tuning. Resume/world/source/optimizer8192/sampler/fixedviews verified;32 native128 PNGs and both dashboard viewports pass. Artifacts retrieved,T4 released. Source4524c5f,archive3c7dd7ba33505d6a5668f79b9d6cdda255c21fccce7dd69a6d9125a9e64b1115. Stop8192; no controller promotion.


§O.32k — current reconstruction continuation
O12|accepted|implementation|Permit32768 total from8192 only; preserve prior envelopes|O11|Resume-step validation; full Python/Ruff; no optimizer/loss/model changes
O13|complete|diagnostic fitting|User32k interpreted32768 total;24576 additional updates|O12|One T4,3000s worker; frozen LeWM/data; saved optimizer/sampler; native128 current-frame MSE
O14|complete|evaluation|Compare8192/32768 fixed current views and train/validation errors|O13|Same controls and image resolution; state/hash verification; retrieve/releaseT4
O.32k.authorization|Explicit user request supersedes8192 cap; stop32768. Hold prediction-specific tuning, collection and controllers.

O.32k.validation|342 local tests pass; Ruff clean; bounded32-step legacy smoke expected warnings. Luna max envelope extension parent-reviewed; freeze before32768 continuation.

O.32k.result|32768 complete;24576 new updates from8192,285.788s fitting,T4 peak192290816B.342 local/123 remote tests, Ruff/smoke pass. Train current MSE0.0000503655 (−93.40% vs8192),changed-current0.00231104 (−94.77%). Validation current0.00335008 (−8.53%),changed-current0.140285 (+0.36%); mean baseline0.00268604/0.137958 still better. Cleaner images, some sharper shapes; object placement inaccurate in held-out views. Resume/optimizer32768/sampler/source/world/fixedviews verified;32 native128 PNGs, both dashboard viewports pass. Artifacts retrieved,T4 released. Sourcea904395,archive92860e4d89a2cc3411eb2d52688badb8d48ea28e22ea4daba4d8f401fa4e33d7. Stop32768; no prediction-specific tuning or controller promotion.


§Q — large practice corpus
Q1|accepted|implementation|Deterministic varied recipes; resumable native capture; bounded-memory dataset; raw CLS reconstruction interface|O14|Parent review; split/corruption/cache tests; native pilot; preserve legacy corpus guards
Q2|complete|collection|4096 train +512 validation episodes ×128 decisions; native128 RGB, cadence4|Q1|Freeze source/recipes; ≤4 CPU workers; no model fitting; stream-validate before READY
Q3|complete|evaluation|Audit counts, episode/recipe separation, hashes, action/family coverage; frame gallery|Q2|Actual totals and unsupported scene types; collection success ≠ learned representation
Q4|authorized|diagnostic fitting|Matched raw CLS versus projected current-frame decoder comparison|Q3|Separate T4 protocol; same corpus/budget; frozen world model within comparison
Q.authorization|User requests massive varied dataset; implementation/collection authorized. Old256-decision importer unchanged; new explicit format. Coordinates/configuration provenance never enters learner inputs.
Q.invariants|Native owns gameplay/actions/terminal flags. Whole unique episode/recipe/seed splits; reject cross-split duplicate captures. Bounded episode cache; no full-corpus pixel/latent cache. Resume only identical plan/source; complete marker after validation. Preserve old runs/checkpoints.

B35|2026-09-15|Large-loader draft accepted ambiguous aliases/dtypes and lacked boundary/recipe checks|Strict new schema; shared cache-miss/stream array validation; Q invariants cover
B36|2026-09-15|Collector draft allowed source drift on resume and stopped on interrupted sidecars|Freeze source/native identity; recover only incomplete episodes; preserve complete receipts; validate split before READY
B37|2026-09-15|Planner draft mixed permanent obstacles into player-only families and repeated one-way stop-go motion|Keep player-only empty; alternate opposing movement bursts; audit native variation

B38|2026-09-15|Full planner rejected odd-size enemy geometry: rounded half-integer margins expand bounds|Use ceil(low)/floor(high); full-plan construction test before capture

Q.validation|356 full-suite tests pass;5 added boundary/publication cases pass (12 loader tests total); Ruff clean; legacy32 smoke pass. Native48-episode pilot6144 transitions strict validation complete; all9 actions in both splits. Full plan4608 unique semantic recipes; source freeze before bulk collection.

B39|2026-09-14|Full corpus correctly rejected identical train/validation bytes; recipe difference only difficulty, invisible in empty arena|Uniform difficulty1; full semantic uniqueness excludes irrelevant metadata; retain pixel-hash rejection. Original v1 remains unpublished. Reopen Q1; audited import of unchanged train capture plus fresh validation into v2, explicit source chain, no silent resume/source mixing.
Q.repair|Reuse only training episodes whose exact native config and capture implementation match new protocol; verify original source hash, unchanged capture AST/dependencies/native binary and every receipt. New v2 records imported provenance; recapture all512 validation at difficulty1. Preserve rejected v1.

B40|2026-09-14|Import pilot passed dataset provenance envelope into frozen capture identity guard|Separate import lineage from exact capture provenance; retain original guard and AST equality

Q.repair.validation|365 full-suite tests pass; Ruff clean; real32-import/16-fresh pilot passes strict publication; original v1 has exactly1 cross-split byte duplicate. Uniform difficulty and semantic uniqueness checks pass; source freeze before v2 import/collection.

Q.result|Published large-practice-20260914-v2:4096train/512validation,589824transitions,594432native128frames;16families,256/32recipes each; all9actions, matched action proportions;64/64 player-start grid cells;0..6enemies,size2..16;39permanent patterns in train. Zero cross-split episode-hash/recipe overlap; difficulty1 both.365 tests,Ruff,legacy smoke; strict streaming validation and64 sampled windows/split pass. Cache4episodes/25366528arraybytes;516096/64512windows. EpisodeNPZ185319858bytes; tar231823360bytes,archive891db46f072dab0390535eac9e7306b0bbcb52374b7899961792365d27c4382b. Manifest683a524eee34030517713d3e29d0f06959c2cf83186313197983b9a9ffef6afa. Source4e38fcc; verified training imports1469782;534.08s v2 phase. Galleries/coverage/checksums retained; Tailscale artifact server8791 verified. Rejectedv1 preserved; no model fitting/controller promotion.
B41|2026-09-14|Gallery midpoint64 aliased repeated scripted motion at128|Use intermediate43; regenerate selected images only; collection/model unchanged


§U — matched current-frame reconstruction on large corpus
U1|complete|implementation|Bounded disk feature/target banks; matched CLS/projected query fitting; T4 launcher and checkpoint/artifact verification|Q3|Unit tests, source review, full Python/Ruff; preserve prior trainers/corpus
U2|complete|feature extraction|Frozen calibrated LeWM;4 sampled frames per episode;16384train/2048validation|U1|One T4; dataset683a524eee34030517713d3e29d0f06959c2cf83186313197983b9a9ffef6afa; pixel targets/masks and features disk-backed; frame choices/index hash frozen
U3|complete|diagnostic fitting|Two fresh QueryPixelDecoder heads; CLS versus projected;512→2048→8192 updates each|U2|Identical init904/sampling903/batch32; AdamW lr0.001 decay0.01; plain current RGB MSE; same minibatches; world state frozen
U4|complete|evaluation|All selected train/validation frames at each milestone; fixedtrainmean and wrong-latent controls; same square images|U3|Both heads finish milestone before evaluation; save optimizer/sampler/RNG; retain checkpoints; retrieve and verify provenance; release T4
U.authorization|User requests training runs; advance implementation→extraction→fitting→evaluation after engineering gates. This first screen follows Q4 and current-frame priority; no LeWM/predictor updates, gameplay collection, or controllers. Cap8192 perhead and one T4 worker7200s. Further world-model training remains separate.
U.invariants|Learner inputs pixels only for current-frame probe; action metadata unused. Validation never fits decoder or mean image. No full-corpus RAM cache. Source/data/world checkpoint frozen; no training-mode BN updates. Checkpoint7714c2afcea1f755513a624a7d0f885361271d4c144869f25c8d3a0189432b5d unchanged. Record measured reconstruction gains separately from gameplay understanding.

B42|2026-09-14|New remote worker first lint classified absent pending module as third-party|Rerun import formatting once module exists; existing Ruff gate sufficient
B43|2026-09-14|Existing dashboard hardcoded next-frame labels would mislabel current-only control images|Explicit current_frame_only metadata selects ordered images and literal labels; legacy snapshots unchanged

U.scoring|Score RGB floats0..1; mean from training only, same scale. Changed mask exact any-channel inequality against preceding native frame, stored by bank. Wrong latent paired from a different episode within same split; never adjacent same-episode sample or cross-split wrap.

B44|2026-09-15|Large-probe draft passed combined train/validation maps into fitting wrapper|Pass train-only maps; end-to-end wrapper fixture asserts optimizer input excludes validation. Existing U.invariants forbid validation fitting; add regression coverage before T4.
B45|2026-09-15|Large-probe draft inferred output batch dimension from one sampled row|Validate decoder shape against current minibatch target; batch-size>1 fixture covers. Existing shape contract sufficient.
B46|2026-09-15|Large-probe draft published latest visualizations only at final milestone|Publish latest plus retained milestone file at512/2048/8192; wrapper fixture covers.
B47|2026-09-15|Tiny wrapper fixture expected16 dashboard views from only8 validation examples after stride2selection|Align fixture family count with intended8train+8validation views; production16families unchanged. Existing bound16 invariant sufficient.

U.validation|372 full-suite tests pass; Ruff clean; legacy32 CPU smoke passes. Parent reviewed bank/fitting/worker; independent Luna audit found no remaining split/scaling/pairing defects. Tiny fixture verifies matched init/sampling, optimizer step, train-only wrapper and early milestone publication. Dashboard JS syntax plus current/legacy metric-label fixture pass; Tailscale8790 serves updated UI. Freeze source before T4 extraction/fitting.

B48|2026-09-15|Colab CPU BLAS differed by1.91e-6 between batched and single-row float32 fake encoder; exact equality blocked remote preflight|Use rtol1e-6/atol1e-5 for computed CLS; retain exact nativepixels/masks and frozen-state checks. First job v1 stopped before extraction/fitting; preserve logs, refreeze corrected tests.
U.preflight_repair|Only computed-feature test tolerance changed;4 bank tests pass and scopedRuff clean. Training/encoding code unchanged from31a46d5. Old T4 terminated; v2 reruns all153 variant tests before extraction.

U.result|lewm-large-probe-20260915-v2 complete; source2870984/archivefd67fa3ab1b5fccff5226a8d64607831c6c24e5469bdecb93c01898ec834d6b3; TeslaT4,torch2.11.0+cu128.153 remote tests pass.16384train/2048validation frozen features;8192updates eachhead; all6 optimizer/sampler checkpoints verified; world SHA unchanged; session released. Final validation MSE CLS0.003013560/projected0.002818151 vsmean0.013009725; changed0.139012175/0.139003738 vsmean0.133331733. Wronglatents multiplyglobalerror2.635/2.804 butchangederror only1.0066/1.0071. Broad reconstruction improves; small moving detail unresolved. All-family/milestone viewer published; no worldmodel/controller update.

§W — balanced bright-pixel reconstruction
W1|complete|implementation|Optional balanced-bright loss; class metrics; matched baseline rescoring; T4 launcher protocol|U4|DefaultMSE unchanged; gradient/empty-class/split tests; parentreview and fullchecks before freeze
W2|active|feature extraction|Same frozen checkpoint, corpus, fourframes/episode and seeds asU|W1|One T4; fresh bank; indexhash must matchU; learner remains pixel-only
W3|pending|diagnostic fitting|Fresh CLS/projected heads;512→2048→8192 each; equal bright/rest class mass|W2|init904/sample903/batch32/AdamW0.001 decay0.01; no worldmodel updates; loss only intendedchange
W4|pending|evaluation|Original global/changed metrics plus bright/background MSE, changed-bright MSE and bright precision/recall/IoU; wronglatents; reevaluate U finalheads on same bank|W3|Recomputed baseline global/changed scores within1e-6 of retained U evidence; retrieve allcheckpoints/results; releaseT4
W.loss|TargetRGB floats0..1; brightmask=min(R,G,B)>=0.8. Loss0.5*meanRGBsquaredError(bright)+0.5*meanRGBsquaredError(rest), per frame then mean across batch; emptygroup uses availablegroup mean for that frame. Target-only mask; no game state/semantic labels. Falsebright predictions on background remain penalized.
W.invariants|Validation never fits or tunes weights. Threshold0.8/masses0.5 fixed before run; identical sample sequence and selectedframes toU. Preserve U artifacts. Ordinary MSE retained for comparison; classweighted loss cannot alone establish improvement. Bright includes HUD and obstacles; not player/enemy identity. One worker7200s,8192updates/head cap; no controllers or futureframe training.
W.authorization|User asks to weight missed white pixels above dominantblue background; implement and run this bounded matched follow-up after engineering gates.

B49|2026-09-15|Baseline delta guard acceptedNaN because NaN>tolerance isfalse|Require finite nonnull metrics beforecomparison; rejection tests cover NaN/±Inf andnull transitions. W4 finite comparison invariant.
B50|2026-09-15|Parent launcher fixture used obsolete unusedroot argument during concurrent helper cleanup|Align fixture with final2argument helper; no productionbehavior change/newinvariant needed.

W.validation|384 full-suite tests andRuff pass. Parentloss/metric/launcher review; tests cover per-frame classbalance, targetmask gradients, emptyclasses, defaultMSE equivalence, classmetrics, null/nonfinite baseline rejection and exactartifactbundling. Updateddashboard5tests plus JSsyntax/weighted/missingclass/legacy displayfixture pass. Freeze before T4.
