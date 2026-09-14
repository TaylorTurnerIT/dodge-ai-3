# Superseded design — reference only

Not controlling instructions. Replaced by [current SPEC](../SPEC.md). No implementation or training occurred under this design.

# Pixel representation DDQN

Status: detailed proposed experiment; code/collection/training unopened. Date: 2026-09-14.
Variant: `pixel-repr-ddqn`. Source namespace: `pixel_repr_ddqn`.
Worktree branch: `experiments/pixel-repr-ddqn`; base `460e662`.
Derived from prior conversation plan, SHA256 `0adc53c7e22036822d8b46fd4e23182f4c6aed81991c54df1178510fdbf9b3f5`. This file is self-contained; earlier worktree is not a runtime or documentation dependency.
Root FORMAT.md absent; existing compact SPEC/table convention retained. User explicitly requested phase kit; this spec defines its authority before kit creation.

§G
G1|Learn useful dense visual/temporal representations from native rendered pixels and executed actions.
G2|Preserve player body/trail distinctions, ordinary/hollow enemies, pattern boundaries, preview-to-obstacle changes; terms describe evaluation questions, never training labels.
G3|Demonstrate scene-dependent greedy avoidance and improved held-out survival with DDQN.
G4|Separate representation quality, controllability evidence, prediction quality, and policy benefit; no combined score hides failure.


§C
C1|Representation inputs/targets: native RGB pixels + executed actions only. Native positions/velocities/entity labels/collision masks/TTC/lookahead/privileged teachers forbidden. Timestamp/lane/episode metadata only sampling/masking/provenance.
C2|Ordinary native reward/termination/truncation allowed for RL targets and boundaries; no new game semantics/reward shaping. Rust remains sole authority.
C3|Native128 RGB; nine actions; fixed4-native-frame decisions; four past/current frames. Future frames only training targets; no future information at action selection.
C4|Train initial teacher/student from Dodge pixels; no external pretrained weights. User screenshot/taxonomy gives evaluation context, never supervision labels.
C5|New variant `pixel-repr-ddqn`; source `src/dodge_native_game/variants/pixel_repr_ddqn/`; tests `tests/variant_pixel_repr_ddqn/`; artifacts `history/dodge/gymnasium/pixel-repr-ddqn/`. No writes to legacy run roots.
C6|Own scope authorized by user; prior CNN-namespace-only constraint applies to legacy variant, not this new variant. Legacy code/spec/config/checkpoints unchanged except additive root routing. Native helper reuse through explicit audited imports; any shared extraction requires separate reviewed spec amendment and parity evidence.
C7|This worktree setup creates documentation/scaffolding only. It does not authorize phase execution, collection, learning runs, servers or remote resources.
C8|Coding phases and collection/training phases separate. Coding phases permit only capped verification; training phases consume frozen code/protocol/data. No opportunistic future-phase work.
C9|Dense representation primary; slots, PPO, external DINO transfer, persistent recurrence, joint encoder/policy finetuning and planners deferred. Open only through subsequent evidence-based spec amendment.
C10|SPEC owns behavior, interfaces, phase state, budgets, invariants and tasks. Kit owns execution order/templates and cites SPEC; no competing acceptance rules. Root historical board/PPO tasks do not select work for this variant.
C11|Numbers are proposed experiment settings, not established optimal choices. P0 fixes protocol choices; P3 calibrates/locks numeric controls before P4 training. Later changes invalidate affected comparisons and require versioned protocol amendment.

§R
R1|ADM|Spatial attention weights local inverse-action logits; self-localization used for exploration bonus, not validated DDQN feature interface|https://arxiv.org/abs/1811.01483
R2|DINO|Teacher/student self-distillation can yield object-like ViT features; semantic role/danger not guaranteed|https://arxiv.org/abs/2104.14294
R3|iBOT|Masked patch-token self-distillation with learned teacher tokenizer; motivates local targets|https://arxiv.org/abs/2111.07832
R4|SPR|Action-conditioned future latent prediction supplies auxiliary RL supervision on Atari|https://arxiv.org/abs/2007.05929
R5|DINOSAUR|Self-supervised features plus object grouping; optional later path|https://arxiv.org/abs/2209.14860
R6|Our adaptation|Full-frame masked patch learning + causal temporal features + inverse/forward branches + frozen-first DDQN; not reproduction of any cited method. Transfer to Dodge unverified.


§I
pixels: proposed `native_adapter.py` → owned lossless128 RGB + nine-action native adapter; reuse audited native boundary, no entity fields exposed.
dataset: proposed `dataset.py` → contiguous lossless sequences + whitelist learner views; content hashes, split IDs, cadence and boundary masks.
collector: proposed `collect.py` → fixed-policy corpus collection, no optimizer; corpus under new artifact root `datasets/<dataset_id>/`.
encoder: proposed `model.py` → Z_t,H_t,alpha_t with new architecture ID; finite-history state only.
representation: proposed `pretrain.py` → student/EMA teacher, inverse/SSL/detail heads; P5 adds prediction branch.
policy: proposed `agent.py`, `replay.py`, `train.py` → isolated DDQN orchestration and temporal raw-pixel replay. Existing `cnn_image_ddqn.agent.DoubleDQNAgent` factory is audited reuse candidate, not automatic dependency on legacy runner.
artifacts: proposed `run_artifacts.py` → new variant path authority; preserve familiar manifest/config/status/metrics/evaluation/report shapes. Legacy writer hardcodes its variant root; do not reuse its path authority blindly.
viewer: proposed `explain.py` → read-only replay-linked diagnostics; no live learner lane access. Server only if later explicitly started; Tailscale link required.
checkpoint: own architecture/config/data/teacher/encoder/source hashes + student/teacher/center/optimizer/scheduler/RNG/counters. Incompatible load rejects before mutation.
kit: `kit/README.md` + `kit/phases/` + `kit/templates/`; checked-in acceptance summaries added only after real phase evidence, never prefilled success.
cli: prospective collection/pretrain/train/evaluate commands are names to implement in respective phases; no entrypoints registered by this scaffolding.

§A — architecture appendix

Symbols: B=batch; T=4 history length; A=9 actions; D=192 feature width; N=32×32=1024 tokens; K=1024 prototypes. o_t=image before a_t; a_t acts until o_(t+1). Z_t=encoder(o_t); H_t=temporal(Z_(t-3:t)); alpha_t=control-related attention, not certified player mask. Teacher quantities barred, stop-gradient.

component|operation|output excluding B
---|---|---
Input|four RGB images, uint8→float /255|4×3×128×128
Frame embedding|nonoverlapping4×4 patches, linear48→192 + fixed2D position encoding|1024×192 per frame
Frame encoder|4 pre-LN Transformer blocks; 6 heads; MLP ratio4; GELU; residuals; final LN|1024×192 per frame
Temporal fusion|same-cell concatenate four Z maps→1×1 conv768→192; two residual3×3 conv blocks at192, GELU; per-location channel LN|192×32×32
Control attention|H_t→1×1 conv192→64→1, GELU hidden; spatial softmax|1×32×32
SSL projection|tokenwise192→256→128→1024; GELU hidden; normalized128 embedding and prototype weights|1024 prototype logits per token
Inverse head|concat H_t,Z_(t+1),Z_(t+1)-Z_t→1×1 conv576→128→9|9×32×32
Forward head|H_t + broadcast embedding(a_t), width192; two residual3×3 blocks→predicted next H|192×32×32
Diagnostic decoder|Z map→3×3 conv192→64→pixel-shuffle4 RGB head; GELU hidden|3×128×128
Policy head|concat H_t and alpha_t;3×3 conv193→64 stride1;3×3 conv64→64 stride2; flatten16384→512 GELU; V512→1, advantage512→9|9 raw Q values

Temporal memory finite-window; no persistent recurrent state. Policy head retains spatial layout; alpha never gates away background/enemies. Frame features shared across times. Four-pixel patches retain all48 raw patch values before projection; token spacing still limits learned localization and requires detail gate. No claim of pixel-precise attention.

SSL: teacher full unmasked frame; student same full frame with random25% patch tokens replaced by learned mask token BEFORE transformer. Loss = cross-entropy teacher/student prototype distributions on masked positions; mean over valid positions/batch. Teacher centering + sharpening; student temperature0.1, teacher0.04, center EMA0.9, parameter EMA0.996 initial. No crop/color jitter/flip/blur initially. Optional global-view DINO loss deferred; call this iBOT-inspired local self-distillation, not full iBOT/DINO reproduction.

Inverse: e_t from before/after features; p(a_t)=softmax(sum alpha_t*e_t). Attention uses causal H_t, not future frame. CE combined + mean per-cell CE -0.001*H(alpha); inverse weights1. Past frames supplied; a_t never input to inverse/attention. This deliberately differs from original current-image ADM attention.

Forward: action-conditioned recurrence over teacher-aligned temporal maps; horizon k in {1,2,4} decisions. Initial H_t real; subsequent predictions consume predictions + recorded intervening actions only. Targets = detached EMA temporal maps at t+k. Normalize features and use mean squared distance across valid tokens/horizons; no future observations in predictor. Full actual future only teacher/training target. P5 implements forward branch after accepted P4 representation; P6 trains it.

Detail loss: decoder reconstructs original RGB from UNMASKED Z; mean absolute error. Weight0.1 initially; target entirely pixels. Auxiliary fidelity pressure, not semantic segmentation supervision. Report whole-frame and image-variance-stratified patch errors; fixed stratification from pixels, no privileged masks.

P4 initial loss: L_SSL + L_inverse +0.1 L_RGB. Ablations use declared subsets. P6 adds 1.0 L_forward. Log per-loss gradient norms/cosines and per-module actual update/weight ratio; weights provisional, not implied equal gradient influence.

Optimizer: AdamW lr1e-4, weight decay0.01 excluding biases/norms; batch16 sequences; clip global norm5; float32 first smoke. Warmup500 updates; seed-specific initialization preserved. EMA parameters/centers update only after valid optimizer step; never during evaluation. DDQN target network separate from EMA teacher. SSL projections/decoder/forward head not required at policy inference.

Implementation precision: every residual conv block = Conv3×3(stride1,pad1)→channel-LN→GELU→Conv3×3(stride1,pad1)→channel-LN, add input, GELU. Policy3×3 convolutions pad1; downsample32→16. Decoder final conv emits48 channels, pixel-shuffle factor4→RGB128. Transformer dropout0 initially. Masked tokens retain their position encoding. No batch normalization.

Teacher initialization: exact student copy; teacher includes frame encoder, temporal fusion and projection; detached parameters. Center vector lengthK; update from uncentered teacher logits over training tokens only. EMA constant0.996 initially; center momentum0.9; bias correction none. AdamW betas(0.9,0.999), eps1e-8; lr linear0→1e-4 over500 updates then constant; AMP disabled for reference smoke. Architecture changes require new ID.

Inverse branch receives UNMASKED causal H_t and unmasked Z_(t+1). SSL masking never substitutes corrupted images into inverse, temporal, RGB detail or policy branches. Future targets not policy inputs. Terminal-transition inverse loss valid only when actual before/after rendered frames exist; forward targets mask unavailable/post-terminal history. Reset-filled histories allowed for acting, excluded from motion-training windows.

Known hypotheses: masked-local self-distillation without original iBOT global loss can collapse; global transformer context means token location need not localize its causal evidence; short-history features can miss long halo timelines. P3/P4 gates inspect these; never describe alpha as a verified player mask or future latents as danger probabilities. Raw feature similarity/attention is diagnostic, not segmentation accuracy.

Forward evaluation precision: candidate and reference latent coordinates can drift. P5 must define a frozen reference teacher and equal-capacity readouts fitted on TRAIN data only for each branch, then freeze them before validation; persistence uses identical reference-space convention. An H decoder is separate from Z diagnostic decoder; never feed H to Z decoder and claim valid RGB prediction. Validate action sensitivity without assuming every enemy depends on our immediate action.


§P — canonical phase map
id|status|mode|goal|depends_on|gate
---|---|---|---|---|---
P0|ready-for-review|design|Lock architecture and experiment contracts|—|Reviewed architecture and protocol draft; no unresolved input-authority or phase-dependency conflicts.
P1|unopened|implementation|Implement pixel sequence and artifact boundary|P0|Deterministic round-trip, lane/episode/future masks, whitelist, namespace and interrupted-write checks pass.
P2|unopened|collection|Collect and freeze representation corpus|P1|Data integrity and coverage accepted; no holdout leakage; explicit coverage gaps; immutable dataset identity.
P3|unopened|implementation|Implement visual and inverse-dynamics learner|P2|Shapes, gradient routing, causality, teacher state, checkpoint resume and resource checks pass; numeric training gates locked.
P4|unopened|training|Train and evaluate visual/control representations|P3|Noncollapse, detail and controllability evidence plus visual review accepted; policy performance not yet claimed.
P5|unopened|implementation|Implement action-conditioned future prediction|P4|Recursive predictions never consume actual futures; frozen-target comparisons and horizon masks tested; P6 protocol locked.
P6|unopened|training|Decide whether prediction adds useful information|P5|Valid fixed-space evaluation and visual evidence; positive result selects predictor-trained encoder, negative result retains P4 base.
P7|unopened|implementation|Implement frozen-representation DDQN and baselines|P6|Correct DDQN target/masks, frozen parameters, artifact ABI and evaluation controls; campaign protocol locked.
P8|unopened|training|Screen policies and lock final comparison|P7|Valid greedy/control evidence; final configuration and independent learner seeds frozen; holdout remains sealed.
P9|unopened|training/evaluation|Confirm frozen choices on fresh seeds and holdout|P8|Complete paired report and uncertainty, scene-dependent greedy behavior and censoring; classify positive/inconclusive/negative.

P0 scope: Review architecture, isolate reuse boundaries, specify splits, budget accounting, gate metrics and resource requirements. Outputs: architecture review; protocol draft; integration map. Implementation owner: —. Successor blocked until §V17/§V25 acceptance; canonical gate above.

P1 scope: New variant data adapter, sequence storage/sampling, split enforcement, artifact writer and bounded collection CLI. Outputs: data implementation revision; tested schema; collection protocol. Implementation owner: P1. Successor blocked until §V17/§V25 acceptance; canonical gate above.

P2 scope: Execute accepted collector; audit coverage from rendered pixels; hash and freeze corpus and disjoint splits. Outputs: dataset manifest/hash; rendered validation clip index; collection report. Implementation owner: P1. Successor blocked until §V17/§V25 acceptance; canonical gate above.

P3 scope: Frame encoder, finite-history fusion, attention, EMA teacher, masked SSL, inverse head, decoder, checkpoints and offline diagnostics. Outputs: representation implementation revision; smoke/timing evidence; P4 protocol. Implementation owner: P3. Successor blocked until §V17/§V25 acceptance; canonical gate above.

P4 scope: Fixed visual/inverse ablations; representation validation and checkpoint choice only. Outputs: ablation report; accepted base encoder/teacher hashes; validation review. Implementation owner: P3. Successor blocked until §V17/§V25 acceptance; canonical gate above.

P5 scope: Latent recurrence, future masks, fixed-target evaluation and persistence/action controls; no policy implementation. Outputs: prediction implementation revision; test evidence; fixed reference/adapter hashes. Implementation owner: P5. Successor blocked until §V17/§V25 acceptance; canonical gate above.

P6 scope: Matched continued base versus base+forward training; choose representation before policy work. Outputs: matched comparison; retained representation hash; explicit prediction decision. Implementation owner: P5. Successor blocked until §V17/§V25 acceptance; canonical gate above.

P7 scope: Frozen encoder integration, spatial dueling head, online/target semantics, raw history replay and matched baseline runners. Outputs: policy implementation revision; regression evidence; P8/P9 protocol. Implementation owner: P7. Successor blocked until §V17/§V25 acceptance; canonical gate above.

P8 scope: Fresh-seed bounded policy screens; inner selection; lock candidate, baseline and confirmation settings. Outputs: inner-screen report; locked candidate/baseline hashes/configs; P9 manifest. Implementation owner: P7. Successor blocked until §V17/§V25 acceptance; canonical gate above.

P9 scope: At least five new matched learner seeds; frozen training recipe; one final holdout evaluation; no tuning. Outputs: final matched report; holdout results; limitations and deferred-extension decision. Implementation owner: P7. Successor blocked until §V17/§V25 acceptance; canonical gate above.

Dependency rationale: data implementation→collection verifies what is recorded before bulk generation; accepted corpus→representation code supplies real shape/coverage contract; representation code→training prevents fixing experiments while learning; accepted representation→prediction isolates predictive benefit; prediction decision→policy code freezes representation choice; policy code→screen prevents learner defects hiding in metrics; screen→confirmation freezes choices before final evidence. Design P0 has no dependency on unimplemented numeric performance measurements; P3 owns their calibration.

§E — execution envelopes

These are maxima proposed for future phase execution, not authorization to run now. Freeze exact device, duration/storage/memory ceiling and protocol before execution. No executable-looking placeholders for unavailable CLI.

mode|owner|per-invocation bound|scientific training allowed
---|---|---|---
Design|P0|read-only inspection and protocol drafting; no collection/optimizer|no
Data implementation verification|P1|<=256 native decisions on fixture seeds; no optimizer|no
Representation/prediction/policy verification|P3/P5/P7|<=32 optimizer updates per smoke invocation; <=256 native decisions when needed; at most3 timing repetitions on named device|no campaign; synthetic/train-fixture only
Corpus collection|P2|initial cap250000 training transitions; separately declared validation collection; exact per-split limits locked P0|no optimizer
Representation screen|P4|2000 updates/condition/learner seed,3 seeds; initial batch16; all controls included in protocol|yes
Representation confirmation|P4|separate accepted screen decision before extension to <=20000 updates/selected condition/seed; no automatic extension|yes, separate run stage
Forward comparison|P6|<=2000 matched additional updates/branch/seed,3 seeds; base continuation versus +forward|yes
Policy screen|P8|<=250000 additional online transitions/condition/seed,3 learner seeds, plus declared common corpus exposure|yes
Policy confirmation|P9|<=1000000 total training transitions per condition/seed including250000 corpus exposure; >=5 new seeds; cumulative all-seed cost explicit|yes

Budget accounting: environment generation cost and per-learner data exposure separately counted; shared corpus does not become free exposure. Validation/evaluation transitions and all optimizer/compute costs separate. Exact episode caps, sample counts, offline-RL update schedule, resampling statistics and device-hour ceiling remain P0 protocol fields; numeric collapse thresholds calibrated P3 before P4. No threshold invention after observing candidate outcomes.

Training phase interrupted for defect: record invalidity, stop relevant job safely, reopen owner P1/P3/P5/P7, spec/backprop if needed, rerun narrow + required regression checks, freeze new revision, reissue protocol/run ID. Preserve unaffected accepted artifacts.

§V
V1|Learner whitelist excludes arbitrary native info. No hidden-state training or evaluation labels; rendered-pixel visual review described as qualitative evidence.
V2|Entire episodes split before windows. Same-lane/cadence history/futures, no reset crossing or lookahead. Terminal versus truncation semantics explicit; bootstrap only according to accepted native contract.
V3|Train/representation-validation/inner-policy/final-holdout game seeds disjoint and immutable; native seeds in0..32767, no clamping. Holdout absent from all pretraining/selection/teacher updates. Learner seeds independently declared.
V4|Control attention interpreted as action evidence; tails, action priors, static-frame shortcuts and boundaries checked. Correct action prediction does not establish body localization.
V5|Representation ablations: random frozen, reconstruction-only, SSL+RGB, inverse+RGB, combined; forward separate P6. Identical corpora/update budgets where trainable, natural and balanced action metrics, label/temporal-shuffle controls.
V6|Feature variance/effective rank, prototype occupancy/entropy, attention entropy, per-loss gradient norms/cosines, parameter update/weight ratios measured. P3 locks calibrated floors using constant/random controls before P4. Low loss alone never learning acceptance.
V7|Detail evaluation covers body/trail, filled/hollow squares, adjacent objects, pattern boundaries and halos on fixed rendered validation clips. Pixel error stratified using pixels only; compare reconstruction-only and constant-image baselines. No semantic segmentation score without labels.
V8|Forward evaluation uses common frozen reference space with train-only frozen readouts where needed; persistence/action-shuffle controls. Moving-teacher loss cannot establish gain. Actual future never fed recursively as predictor input.
V9|No exploration bonus, new game semantics or reward changes. Masking limited to representation SSL branch; actual policy RGB intact. No crop/color/flip augmentation in initial candidate.
V10|Initial DDQN freezes encoder/temporal/attention and trains only policy head. Complete Q-target copy/sync separate from EMA teacher; no optimizer state can update frozen features. End-to-end baseline deliberately separate.
V11|Baseline suite includes current RGB DDQN, proposed architecture without pretraining, frozen random representation. Equal total transition exposure including pretraining; shared corpus also available to baselines under predeclared offline/online schedules. Report optimizer updates/pretraining and online compute separately.
V12|Final comparison uses >=5 fresh matched learner seeds beyond screen seeds. Freeze difficulty/patterns/reward/cadence/evaluation cap and checkpoint selection rule. Greedy neutral/fixed/random controls, per-seed survival/action counts, Q-gap scene response and censoring reported.
V13|Positive policy claim requires positive95% paired CI on mean survival difference versus strongest predeclared baseline, plus scene-sensitive greedy behavior; uncertainty method accounts for learner seeds, not frames as independent samples. Otherwise inconclusive/negative.
V14|Phase-specific code/protocol/data/config hashes immutable during runs. If implementation changes, stop affected run, retain evidence, reopen owning coding phase, reverify and create new run/protocol. No patch-and-resume under old scientific identity.
V15|Checkpoint restores student/teacher/center, all active optimizers/schedulers/RNG, data position and counters; fixed-sample resume equivalence. Evaluation cannot update model/EMA/centers/RNG used for training.
V16|New variant imports/startup never change legacy defaults, entrypoints, checkpoints or artifact roots. No namespace promotion via editing legacy SPEC. New architecture/ABI rejects incompatible state before tensor mutation.
V17|Phase status distinguishes unopened/active/implementation-complete/evidence-ready/accepted/rejected/inconclusive. Accepted phase != positive scientific result. Only explicit acceptance or already-authorized auto-advance opens successor; failed prerequisite never bypassed.
V18|Implementation closures run meaningful narrow tests plus `scripts/uv-run pytest`, `scripts/uv-run ruff check .`, unique32-step legacy CPU smoke and applicable new-variant capped smoke. Collection/training closures verify hashes and evaluate evidence, not write implementation.
V19|Root SPEC links scoped authority. New variant SPEC owns all phase/task/bug IDs; kit links only. No fake acceptance reports, fabricated metrics, runnable-looking unimplemented CLI commands or completed task marks.
V20|Representation data collection P2 has no optimizer. Representation training P4/P6 contains no policy optimization. Policy P8/P9 uses frozen selected representation. No campaign launched in coding phase beyond §E verification caps.
V21|P2 incomplete scene coverage blocks P4 semantic claims; amend collection protocol and rerun P2 through P1 if collector changes. Standard challenge scenes/patterns retained; easy-mode-only result not final capability.
V22|P6 valid negative result may accept decision to retain P4 model; invalid implementation or failed P4 representation cannot advance. P8 negative screen yields no-go confirmation unless documented scientific reason accepted.
V23|P9 holdout opens once after recipes and checkpoint hashes frozen. No tuning on holdout; later exploration requires new experiment and untouched holdout. Preserve all seed results, no best-seed-only claims.
V24|Servers, GPU allocation and external jobs require explicit phase execution scope; protect all existing processes/worktrees. User-requested scaffolding never implies long-run authorization.
V25|Proof of phase gate includes commands, return codes, device/runtime/source hashes, artifact paths, limitations, visual review and acceptance authority. Failures considered for §B backprop; do not weaken gates to advance.

§T
id|status|task|cites
---|---|---|---
T1|.|P0: Inspect native RGB, action, episode and reward boundaries; audit proposed imports without modifying legacy variants.|C8,C10,V14,V17,V25,I.kit
T2|.|P0: Resolve protocol choices: sample units, learner seeds, split rules, budget accounting, uncertainty statistic, validation criteria.|C8,C10,V14,V17,V25,I.kit
T3|.|P0: Publish review verdict and explicit unresolved numeric calibrations for P3; hand off only data implementation.|C8,C10,V14,V17,V25,I.kit
T4|.|P1: Implement owned RGB/action sequence storage and versioned artifact paths under new namespace.|C8,C10,V14,V17,V25,I.kit
T5|.|P1: Test reset/terminal/truncation boundaries, forbidden info exclusion, exact palette round-trip and dataset split assignment.|C8,C10,V14,V17,V25,I.kit
T6|.|P1: Run bounded fixture/native checks; freeze collector revision and P2 protocol; publish acceptance handoff.|C8,C10,V14,V17,V25,I.kit
T7|.|P2: Verify collector revision, clean status and protocol hash; collect only declared game-seed pools and transition budget.|C8,C10,V14,V17,V25,I.kit
T8|.|P2: Check sequence integrity and rendered coverage of tails, hollow enemies, adjacent enemies, pattern edges and previews.|C8,C10,V14,V17,V25,I.kit
T9|.|P2: Freeze dataset and split hashes; publish evidence and next-phase handoff; do not train any model.|C8,C10,V14,V17,V25,I.kit
T10|.|P3: Implement appendix components except forward predictor and policy head; add shape, teacher and sequence tests.|C8,C10,V14,V17,V25,I.kit
T11|.|P3: Build synchronized offline diagnostic output; calibrate collapse/detail controls without candidate training, then freeze P4 thresholds.|C8,C10,V14,V17,V25,I.kit
T12|.|P3: Run capped CPU/GPU smoke and resume tests; freeze code, configuration, resource cap and training protocol.|C8,C10,V14,V17,V25,I.kit
T13|.|P4: Verify code/data/protocol hashes; run only named pretraining ablations with matched budgets.|C8,C10,V14,V17,V25,I.kit
T14|.|P4: Evaluate fixed controls, action confusion, per-loss gradients and unseen validation clips; distinguish tail attention from body localization.|C8,C10,V14,V17,V25,I.kit
T15|.|P4: Choose base representation using representation-validation only; publish accepted/negative/inconclusive result and preserve all runs.|C8,C10,V14,V17,V25,I.kit
T16|.|P5: Implement action-conditioned recurrence and horizon masks against detached EMA targets.|C8,C10,V14,V17,V25,I.kit
T17|.|P5: Implement common frozen-reference evaluation, persistence and action-shuffle controls; use a trained/frozen adapter when feature coordinates differ.|C8,C10,V14,V17,V25,I.kit
T18|.|P5: Run capped smoke; lock forward-loss experiment and reference-space protocol; publish handoff.|C8,C10,V14,V17,V25,I.kit
T19|.|P6: Start matched branches from same P4 checkpoint; match new samples and optimizer budget.|C8,C10,V14,V17,V25,I.kit
T20|.|P6: Compare fixed-reference prediction, persistence, action sensitivity and visual retention; moving-teacher loss alone is insufficient.|C8,C10,V14,V17,V25,I.kit
T21|.|P6: Publish decision; negative scientific result may retain P4 base, implementation invalidity may not bypass gate.|C8,C10,V14,V17,V25,I.kit
T22|.|P7: Implement finite-history policy adapter and dueling head; preserve separate EMA-teacher and Q-target identities.|C8,C10,V14,V17,V25,I.kit
T23|.|P7: Implement/tests for matched RGB baseline, random representation and end-to-end architecture controls; same data access and accounting.|C8,C10,V14,V17,V25,I.kit
T24|.|P7: Run capped new/legacy smokes and required regression checks; freeze screen protocol and confirmation decision rules.|C8,C10,V14,V17,V25,I.kit
T25|.|P8: Run only approved screens with equal total transition exposure and declared offline/online schedules.|C8,C10,V14,V17,V25,I.kit
T26|.|P8: Evaluate inner greedy behavior versus neutral/fixed/random controls; include complete episodes, censoring and Q-gap scene response.|C8,C10,V14,V17,V25,I.kit
T27|.|P8: Lock strongest declared baseline and candidate using inner evidence; publish go/no-go for confirmation without inspecting holdout.|C8,C10,V14,V17,V25,I.kit
T28|.|P9: Verify frozen recipes and unused confirmation learner seeds; train candidate and baselines under matching budgets.|C8,C10,V14,V17,V25,I.kit
T29|.|P9: Select checkpoints by predeclared rule; freeze hashes before final holdout opens; run matched evaluation to declared cap.|C8,C10,V14,V17,V25,I.kit
T30|.|P9: Publish all per-seed results and paired confidence intervals; no extension or parameter tuning after result.|C8,C10,V14,V17,V25,I.kit

§B
id|date|cause|fix
---|---|---|---
