# CNN image Double-DQN variant

Isolated major-model variant. Native Rust remains the only source of game
transitions, collision geometry, rewards, and terminal state.

§G
G1|Train CNN DDQN from full native rendered RGB pixels; collision-image runs retained as explicit baseline.
G2|Version observation profiles; new pixel experiments use `native-rgb-v1` + four temporal frames.
G3|Use a standard Atari-style convolutional trunk with a dueling Double-DQN head.
G4|Keep this experiment isolated so future model variants cannot silently share
   observation or training assumptions.
G5|Launch bounded runs and show live metrics/controls in LaunchSpark-parity Pygame dashboard.

§C
C1|`collision-image-v1` excludes rendered effects; `native-rgb-v1` uses full native framebuffer including particles, UI, palette + camera; no cropping/resizing/masking.
C2|Collision native raster `u8[84,84]`; RGB native framebuffer palette expands losslessly to `u8[3,128,128]`; no Python game semantics.
C3|Legacy default stays collision `float32[N,84,84]`; explicit RGB profile `uint8[3N,128,128]`, oldest→newest RGB triplets; normalize once at model boundary; stack length configurable.
C4|Existing nine native actions and default `step_frames=4` remain unchanged.
C5|Native reward/termination are passed through; Python adds no game semantics.
C6|Initial learner is image-only; scalar/state MLP inputs are a later variant.
C7|Dashboard/run artifacts required; local pause/save/load/config controls only; HPO and broad campaign claims stay out of scope.
C8|Watch Agent opens a separate read-only browser replay generated from a saved `.pt` checkpoint.
C9|Approved reward experiments may extend native Rust event/reward boundary; policy remains fullnative pixels only; legacy reward/defaults unchanged; Python weights only.

§I
native: `collision-image-v1` → deterministic grayscale collision raster
native: `native-rgb-v1` → native 128×128 framebuffer, lossless PICO-8 palette → RGB; particles + HUD retained
cli: `--observation-profile collision-image-v1|native-rgb-v1` → same profile in train/eval/checkpoint/replay; missing legacy metadata means collision only
cli: `--observation-profile native-gray-v1` → fullnative128 luma uint8 stack; `--n-step 1|3` → collision return horizon; default1 unchanged
env: `CollisionImageEnv` → Gymnasium `Env`, `Discrete(9)` → `Box(0,1,(N,84,84))`
stack: `FrameStack` → exactly N newest native frames, default N=4
model: `CNNQNetwork` → dueling Q-values, no softmax
learner: `DoubleDQNAgent` → online/target networks, replay, Huber loss
reward-foundation: native `PowerupCollected` bit5 + `powerups_collected` uint32 lane counts; `reward_boundary_costs` → validated edge/corner costs; no active training-profile change
artifacts: `RunRecorder` → manifest, atomic status, metrics, evaluation, report
dashboard: `TrainingDashboard` → LaunchSpark-parity Pygame charts/modals/controls over `DodgeDDQNSession`
 replay: `native_replay` → fresh native Rust lane + collision images → bounded `ReplayStore` → Tailscale browser page
 compare: `run_replay` → run's own checkpoint + game settings + frozen inner/holdout seeds → labeled best/median/worst `NativeReplay` set → side-by-side `/compare` page

§M
profile: declarations below = collision84; RGB128 uses input `(B,3N,128,128)`, sameconvkernels → `(B,64,12,12)`, flatten9216→shared512
input: `(B,4,84,84)`
conv1: `Conv2d(4,32,8,4)` + ReLU → `(B,32,20,20)`
conv2: `Conv2d(32,64,4,2)` + ReLU → `(B,64,9,9)`
conv3: `Conv2d(64,64,3,1)` + ReLU → `(B,64,7,7)`
head: Flatten 3136 → Linear 512 + ReLU → value(1) + advantage(9)
combine: `Q=V+(A-mean(A))`
target: `argmax_a Q_online(next_state,a)` then `Q_target(next_state,argmax_a)`

§L
L1|Reset native lane and fill the temporal deque with the initial collision image.
L2|Encode the deque as channel-first float32 values in [0,1].
L3|Select epsilon-greedy action from online Q-values.
L4|Advance native lane exactly one decision interval and append returned image.
L5|Store `(state, action, reward, next_state, done)` as uint8 image frames in replay.
L6|Normalize uint8 images to float32 only for the model forward/update path.
L7|Optimize online Q against Double-DQN target; clip gradient norm at 10; sync target periodically.
L8|Evaluate with exploration disabled and fixed held-out seeds.
L9|Write manifest/status/metrics/checkpoint/evaluation artifacts for every bounded run.
L10|Dashboard queues pause/save/load/game-config requests; learner applies them only at safe points.
L11|Watch Agent loads the newest checkpoint in a separate worker, greedily replays a fresh native lane, and opens its browser URL.

§V
V1|Collision profile never reads rendered pixels; RGB profile reads native framebuffer only, never collision geometry or hidden-state features.
V2|Every native collision image is exactly 84×84, finite, deterministic, and bounded.
V3|Frame stack contains exactly N frames in oldest→newest channel order.
V4|Reset fills all N temporal slots with same initial frame; each step appends one frame (RGB triplet for RGB profile).
V5|Env action/reward/termination/step cadence match the native boundary.
V6|Same seed + same action trace produces identical image hashes and stacked arrays.
V7|Collision CNN retains declared 84×84 architecture; RGB128 uses same conv kernels/strides, 64×12×12 final maps + shared512; input12 channels for stack4.
V8|Q output is raw action values; no softmax or probability normalization.
V9|Default head is dueling: `Q=V+(A-mean(A))`, with one scalar V and one A per action.
V10|Double-DQN target selects next action with online Q and evaluates it with target Q.
V11|Terminal transitions zero the bootstrap term; nonterminal transitions bootstrap.
V12|Replay stores owned uint8 frames and converts/normalizes only at update time.
V13|Every optimizer update clips gradient norm to at most 10.0.
V14|No batch normalization, max-pooling, or spatial augmentation; RGB input exclusive to explicit RGB profile.
V15|Replay samples have validated shapes/dtypes and do not alias mutable storage.
V16|Changing stack length changes only observation channel count, not native state/action semantics.
V17|Variant code and artifacts remain under this variant namespace.
V18|Every run has an immutable manifest, atomic status, append-only metrics, and final report or explicit failure.
V19|Malformed/incomplete runs are visible as invalid; dashboard never presents them as passing.
V20|Dashboard renderer copies telemetry under lock, then renders outside lock; control requests never mutate mid-step.
V21|Live environment enables native payload for selected profile; unavailable payload errors, never substitutes another view.
V22|ReplayBuffer.sample preserves configured `(C,H,W)` shape through ReplayBatch + agent; supported spatial sizes 84×84 and128×128.
V23|Dashboard controls save/load native `.pt` checkpoints and persist reward/game panel values without bypassing artifact contract.
V24|Pygame remains the local LaunchSpark control window; Watch Agent uses a separate browser/Tailscale replay transport.
V25|CPU trainer disables NNPACK before first CNN forward when using native CPU backend; bounded run publishes terminal artifact.
V26|Browser replay code never shadows the training `ReplayBuffer`; both replay surfaces import and construct independently.
V27|Replay uses checkpoint's declared observation profile from fresh native lane; live trainer lane/control queue never shared.
V28|Replay output is bounded to at most 600 decisions and 8 in-memory replays; HTTP routes expose only generated tokens, metadata, and PNG frames.
V29|Watch Agent opens the generated Tailscale URL when possible and reports the URL in the dashboard event stream when browser launch is unavailable.
V30|Every log row carries reward mix, TD error mean/std, action balance/counts, dead-unit share, and effective epsilon schedule; no Python game semantics.
V31|Epsilon decay uses configured global environment-step schedule; bounded smoke exploitation requires explicit short decay, never implicit per-segment rescaling.
V32|Final evaluation uses frozen inner (offset 10_000) and holdout (offset 20_000) seeds plus a forced-action counterfactual on the least-used action; train/holdout gap is reported.
V33|Quality gate is computed from warmup/target/decay/sparsity/lock-in/gap checks; bounded runs stay `warn` and `pass` requires 5_000+ steps with no warnings.
V34|Forced-action replay reuses a fresh native lane and greedy policy except at forced indices; it never touches the live trainer lane.
V35|Run comparison replays pool only inner/holdout greedy-policy episodes; best/median/worst rank by (reward, survival, seed) and each replay carries its run/label/source/eval-reward tag.
V36|Every comparison replay reuses the run's own checkpoint and game settings and runs to termination within the 600-decision bound; the compare page never invents seeds or settings.
V37|Replay builds the Q-head matching the checkpoint (dueling vs plain `q_head`); a plain-head checkpoint replays instead of failing load.
V38|The reported best is the best greedy-policy episode of the frozen final evaluation; the training-curve maximum stays visible only as labeled training context, never as the run's best.
V39|Run comparison serves plotted context per run: the downsampled training reward curve with its exploration-era max marked, plus the final-eval inner/holdout dot strip with best/median/worst ringed.
V40|Target sync interval unit = optimizer updates; metrics + gate use actual sync count, not environment-step inference.
V41|∀ run segment → local + global environment steps explicit; resume restores global epsilon progress while replay/RNG/env reset declared.
V42|Manifest records source revision/dirty state, initialization ID, parent checkpoint SHA-256, parent step, + resume mode; unavailable fields explicit.
V43|Metrics preserve every completed episode return exactly once; partial episode accumulator never labeled completed return.
V44|Counterfactual baseline + forced action use identical seeds; artifact stores per-seed paired deltas + mean delta.
V45|Evaluation records termination/censoring per seed + censored share; censored evaluation cannot support pass gate.
V46|∀ logged optimizer diagnostic sample → target mean, Q-target mean, pre-clip gradient norm, + clipped flag logged.
V47|Action balance over declared action space includes zero-count actions; metrics separate cumulative behavior counts from recent greedy-policy counts.
V48|Seed diversity experiment separates learner RNG seed from nested game-seed pools of 700/5000; pool ∈ [0,9999], eval ∈ [10000,32767], no overlap/clamping; natural episode boundaries only.
V49|Pool membership immutable in manifest; completed episodes record game seed; per-seed training-step counts sum to segment steps; configured pool size ≠ measured visited count.
V50|Requested intermediate snapshots retain global checkpoint steps; evaluation after training avoids changing training RNG stream.
V51|Explanation input + Q/V/advantage captured before action; reward/events/termination labeled after action; immutable checkpoint SHA-256 + native config retained; no live training state.
V52|Dueling explanation Q equals forward output; V equals mean Q; pairwise 512-feature contributions + bias reconstruct fixed-action Q gap; plain head has no invented V.
V53|Blur sensitivity = original minus perturbed V or fixed chosen-alternative Q gap; all-stack + single-frame modes explicit; sign, mask/grid, scale + sensitivity limitations visible; no model mutation.
V54|Explanation replay pause/scrub aligns game image, exact oldest→newest input stack, predictions + timelines; native events unavailable → unknown, never fabricated zero.
V55|Offline feature/channel examples use real captured observations; channel suppression changes inference activation only; activation ≠ importance; traces ≤4096 decisions each, explanation cache + inference concurrency bounded.
V56|Realtime replay awaits decoded native image before advancing aligned input/Q cursor; slow loads never repeatedly blank/cancel image; pause/seek/episode switch invalidate pending playback; native cache bounded.
V57|Viewer uses viewport panes + pagination, preserves access to replay/analysis controls without document scrolling at 1280×720 and 1366×768.
V58|Each input-stack tile/selector labels native frame; single-frame default newest; historical input warns when compared with current native image; reset-padding labels initial native frame.
V59|RGB input round-trips exactly to native display palette, 128×128, includes effects; reset fills temporal slots, each step appends RGB triplet; same actions preserve native reward/done/frame.
V60|Observation profile + spatial shape + temporal depth immutable in config/manifest/checkpoint; mismatched resume/replay rejects; legacy missing profile resolves collision only.
V61|RGB replay stores lossless packed native palette frames + temporal references, ≤2GB at100k transitions/stack4; no duplicated RGB stacks; long run requires measured RAM + add/sample throughput + native parity; no silent capacity reduction.
V62|Pixel experiments use fresh learner, frozen disjoint train/inner/holdout seeds, unchanged reward/game/cadence; retain10k/200k/500k checkpoints; high-score claims include held-out distribution + censoring, not selected maximum alone.
V63|Expensive optimizer diagnostics sampled at logging cadence; unsampled updates transfer no diagnostic scalars toCPU and preserve identical weight updates; sampled vector ≤1 CPU transfer; metric sample step explicit; GPU throughput/telemetry overhead measured before long-run promotion.
V64|Packed replay reconstructs exact chronological RGB stacks across identical consecutive images, terminal/reset padding + ring wrap; ≤2 newframe nodes pertransition; reference tests capacities1/2/7, stacks1/4/8, repeatedcolors + shortepisodes.
V65|Training samples packed palette frames; palette expansion + normalization on training device, no CPU RGB minibatch expansion; decoded inputs and DDQN updates equal denseRGB reference; timings include device decode.
V66|Routine telemetry promotion budget ≤5% added wall time vs minimal logging on matched GPU training workloads; ≥3 alternating paired trials, same device/config/seed/update count; report training-loop and all-in times separately; expensive explanation inference offline only; microbenchmarks alone cannot pass gate.
V67|Three-step return sums gamma^i reward; bootstrap gamma^k at actual suffix length; terminals zero bootstrap, truncations flush but bootstrap; no cross-episode samples; n=1 behavior unchanged; native reward semantics unchanged.
V68|Grayscale derives full native framebuffer via fixed integer luma (299R+587G+114B+500)//1000; uint8[N,128,128], no crop/resize/mask; packed palette replay exactluma +≤2GB at100k; RGB/collision defaults unchanged.
V69|T4 branches freeze700gamepool, lr1e-4, warmup20k, epsilon500k, sync10000, update4, replay100k; compare nstep3/gray/learner43+44 at200k with10k snapshot; 10k explicitly untrained; fresh frozeneval base512,128episodes,4096decisioncap; active A100 untouched.

V70|Pickup event emitted exactly once for collected personality2/3/4; explosion destroying other powerups emits no pickup; ordered counts survive batch/PyO3; existing event bits unchanged.
V71|Native boundary field bounded/symmetric/flat interior; narrow shallow edge term + wider corner overlap; finite validated inputs; no center bonus; geometry tests cover monotonicity and perimeter route.
V72|Reward experiments keep survival scale, count terminal penalty once, separate component totals + immutable weights; no live mutation; pixel observations unchanged; promotion requires matched evaluation +≤5% telemetry overhead.
V73|Multiple native Death events from overlapping hazards in one decision → one life penalty; positive pickup/destruction multiplicity retained; no game transition changes.
V74|Epsilon-greedy samples exploration before Q inference; fully random action skips online forward while preserving observation validation + seeded action trace.
V75|Evaluation decision uses one online-network forward for Q values, greedy action + dead-unit probe; reported values match separate reference calls.
V76|Native pixel replay accepts owned palette-ID frames from native result; sampled RGB/gray tensors exact current reconstruction + no mutable-storage alias.
V77|Opt-in multi-lane collector counts `steps` as exact native transitions; epsilon indexed pertransition; optimizer cadence remains transition-based; lane1 native trace exact; each lane owns seed/episode/temporal/replay chain across terminal reset.
V78|Batched evaluation freezes inactive native lanes + preserves seed order, rewards, survival, actions, termination, Q spread + dead-unit aggregate vs serial reference.
V79|Learner backend explicit in config/checkpoint context; CUDA-only modes rejectCPU; AMP unscales before gradclip; fusedAdam/channels-last/compile/nonblocking transfer never alter eager checkpoint keys; baseline behavior unchanged.
V80|Training benchmark reports native transitions/s + optimizerupdates/s with explicit training/all-in denominators, matched config/seed, alternating order, ≥3 trials; CPU result ! GPU claim; performance ! learning quality.
V81|Multi-lane nstep owns one accumulator/lane; terminal/truncation/final-segment flush emits each transition once, preserves gamma^k suffix, never crosses lane or episode.

§T
id|status|task|cites
---|---|---|---
T1|x|Expose native `collision-image-v1` through batch + PyO3 boundary|C1,C2,C5,V1,V2
T2|x|Implement `CollisionImageEnv` and configurable `FrameStack`|C2-C5,V3-V6,V12
T3|x|Implement CNN trunk, dueling head, uint8 replay, and Double-DQN update primitive|M,V7-V15
T4|x|Integrate env + agent smoke episode and deterministic trace test|V5,V6,V10,V11,V17
T5|x|Add run artifacts and LaunchSpark-parity Pygame dashboard/session adapter|G5,C7,V18-V20
T6|.|Run stack-length ablation harness for N=1,2,4,8|G2,C3,V16,V17
T7|x|Run fixed-seed image-only baseline and publish metrics visible in dashboard|G1,G2,G5,C7,V6,V17-V20
T8|x|Add bounded native replay generator, Tailscale HTTP page, and Watch Agent pop-out|C8,L11,V26-V29
T9|x|Add reward-mix/TD/action-balance/dead-unit diagnostics, effective epsilon schedule, frozen train/holdout eval, counterfactual probe, and computed quality gate|V30-V34
T10|.|Add run-anchored best/median/worst replay comparison over the run's own checkpoint, settings, and eval seeds with a side-by-side browser page|C8,V26-V29,V35-V39
T11|x|Fix DDQN measurement integrity: sync units, resume/provenance, completed episodes, paired counterfactual, censoring, optimizer metrics, action coverage|V40-V47
T12|x|Add nested game-seed pools, measured exposure, and 10k/200k checkpoints for 500k A100 comparison|V48-V50
T13|x|Add checkpoint-backed held-out explanation replay, signed perturbations, feature examples + channel ablations|V51-V55
T14|x|Fix slow-network native playback + fit explanation viewer to viewport|V54-V57
T15|x|Label temporal input timestamps + default same-time comparison after reported enemy displacement|V54,V58
T16|x|Add full native RGB profile +128 CNN/replay shape support; gate exactpixel/nativeparity + legacytests before trainingintegration|V1,V3-V9,V14-V17,V21,V22,V59
T17|~|Wire profile through CLI/train/eval/checkpoint; gate bounded CPU update/checkpoint/native rollout before remote launch; RGB inspector expansion deferred by user|V18,V27,V59,V60
T18|~|Run bounded nativeRGB10k/200k/500k target1000vs10000 experiment on A/H Colab, collect artifacts + paired baseline report; depends T16,T17 gates|V48-V50,V61,V62
T19|~|Remove per-update diagnostic transfers; sample at log cadence + benchmark overhead separately before remote promotion|V46,V63,V66
T20|x|Add opt-in three-step collision returns + handcomputed terminal/truncation tests|V10,V11,V63,V67
T21|x|Add fullnative grayscale profile + exact packed luma parity tests|V59,V60,V61,V68
T22|~|Deploy bounded T4 screens nstep3, grayscale + learner43/44 replicas; collect before release; queue under quota|V48,V49,V69
T23|x|Native reward foundation: exact pickup count boundary + tested edge/corner field; preserve game/RNG/pixel parity|V5,V6,V70,V71
T24|~|Wire versioned native reward components + weights/train/eval/resume contract; death/pickup/destruction isolated treatments; no default change|V18,V60,V63,V72
T25|.|Broaden screenshot-only diagnostic corpus; integrate learning gate + inner checkpoint selector; run independent architecture/reward screens after correctness gate|V47,V55,V62,V66,V72
T26|x|Remove redundant action/eval inference + native-pixel replay conversion/copies; benchmark bounded trainer path|V12,V15,V31,V59,V63-V66,V74-V76
T27|x|Add active full-native batch boundary, multi-lane temporal collector + stream-safe packed replay, exact transition accounting, and batched frozen evaluation|V5,V6,V12,V15,V31,V43,V48,V59,V77,V78
T28|x|Add explicit AMP/compiled fused CUDA learner backend + matched bounded training benchmark; retain eager baseline + artifact compatibility|V13,V18,V42,V63,V66,V79,V80
T29|x|Extend multi-lane collector to independent three-step returns + flush every lane at bounded segment end|V67,V77,V81

§B
id|date|cause|fix
B1|2026-09-10|Live adapter test reused a native boundary with collision-image disabled, leaving an optional field as `None`|Enable `collision_image=True` in the live integration and keep V21 explicit
B2|2026-09-10|PyO3 payload test invoked Cargo with a Python interpreter that had no NumPy module|Keep Rust unit tests independent of the NumPy capsule; verify the actual NumPy payload through the Python boundary test
B3|2026-09-10|Relative `PYO3_PYTHON` was resolved from Cargo's native working directory and did not exist during the attempted workaround|Drop the fragile interpreter override from the native Cargo command
B4|2026-09-10|ReplayBatch retained a fixed four-channel validator after ReplayBuffer became configurable|Validate generic `(N,C,84,84)` batches, carry the configured shape, and enforce it at the agent boundary; V22
B5|2026-09-10|HTTP read-only dashboard did not match requested LaunchSpark Pygame controls/layout|Copy upstream renderer/series/telemetry/reward modules; replace backend with native DDQN session and safe-point control plane; V20,V23
B6|2026-09-10|PyTorch CPU NNPACK probe emitted unsupported-ISA warnings then terminated bounded run with SIGILL|Disable NNPACK before CPU learner construction and cover helper with test; V25
B7|2026-09-10|Browser generator was first added as `replay.py`, colliding with the existing training replay buffer|Keep browser replay in `native_replay.py` and assert both module surfaces remain importable; V26
B8|2026-09-11|Smoke runs stayed near epsilon 1.0, logged no sparsity/TD/coverage signals, and had no frozen holdout or lock-in check|Cap decay at run length, log diagnostics per row, freeze inner/holdout/counterfactual eval, compute gate; V30-V34
B9|2026-09-11|Replay builder assumed a dueling head, so plain-head checkpoints failed load with missing `value_stream` keys|Detect the head from checkpoint keys and build the matching network; V37
B10|2026-09-11|Training-curve maximum was reported as the run's best game, but it came from an exploration-era episode of an unsaved policy|Report the best frozen final-eval episode instead and plot training max as labeled context; V38,V39
B11|2026-09-11|Gate compared optimizer-update target interval with environment steps → zero-sync runs escaped warning|V40
B12|2026-09-11|Resume reported segment steps as total + restarted epsilon from local step without declaring replay/RNG/env reset|V41,V42
B13|2026-09-11|Counterfactual used different seeds from greedy baseline → forced-action effect confounded by scenario|V44
B14|2026-09-11|Periodic `reward` mixed partial + completed episode accumulators and omitted completions between log boundaries|V43
B15|2026-09-11|Action balance dropped zero-count actions + cumulative epsilon exploration masked greedy lock-in|V47
B16|2026-09-11|Resume setup failure occurred after status became running but before the failure boundary, leaving a stale running artifact|Wrap setup/load in failure-state handling; V18
B17|2026-09-11|One learner initialization seed was mistaken for one game seed despite per-episode seed increments; unbounded increments can enter eval ranges|Separate learner/game seed semantics and reserve disjoint pool; V48,V49
B18|2026-09-11|Explanation server draft exceeded Ruff line limit|Format draft before verification; mechanical failure, no new invariant
B19|2026-09-11|Explanation draft suppressed input-stack channels instead of final convolution channels + exposed perturbed gap as signed sensitivity|V53,V55; test conv-channel 63 ablation against manual zeroing + signed fixed-pair map
B20|2026-09-11|UI draft converted null V to zero, counted false event flags as events, and requested heatmaps during playback|V52,V54,V55; Node contracts preserve null + presence semantics and forbid playback inference; DOM stub extended for rendering
B21|2026-09-11|Every playback tick hid native image + replaced pending src before slow load completed → native view only appeared when paused|V56; delayed-image regression + browser playback check
B22|2026-09-11|Native action replay test line exceeded Ruff limit|Wrap zip arguments; mechanical failure, no new invariant
B23|2026-09-11|First viewport draft clipped timeline + left core replay controls in scrolling cards|V57; inspect actual element bounds + panel overflow, not document height alone
B24|2026-09-11|Viewport initialization needed DOM APIs absent from Node stub|Extend stub + tab/pagination contracts; playback-only harness excludes layout startup
B25|2026-09-12|Oldest input default compared native193 with current205 → apparent enemy displacement|V58; newest default + temporal labels; matching newest screenshot aligns enemies
B26|2026-09-12|Telemetry test imported nonexistent top-level tests package|Use relative sibling import; mechanical harness failure, no new invariant
B27|2026-09-12|Agent extracted8 CPU diagnostic scalars on everyoptimizerupdate despite sparse logging|V63; sampled diagnostics + onevectortransfer + exactweightparity tests
B28|2026-09-12|Benchmark draft exceeded Ruff line lengths; import classification pending newmodule creation|Format ownfiles + rerun lint afterintegration; no new invariant
B29|2026-09-12|Concurrent test import observed partial run.py edit|Wait module handoff before integrated verification; harness failure, no new invariant
B30|2026-09-12|Packed replay draft reused frameID for identicalimage and erased elapsed temporal position|V64; alwaysappend chronological node; exactreference wrap/reset/repeat tests
B31|2026-09-12|CPU screen measured packedRGBsample32 108.74ms vsdense18.25ms despite memorysaving|V65; retainpacked minibatch throughCPU sampling + devicepalette expansion; rerunperformancegate
B32|2026-09-12|Integrated telemetry test missing blank line between absolute and relative imports|Ruff import fix; mechanical failure, no new invariant
B33|2026-09-12|RGB campaign draft exceeded Ruff line limit|Wrap literals + comprehensions; mechanical failure, no new invariant
B34|2026-09-12|T4 screen draft protocol literal exceeded Ruff line limit|Shorten literal; mechanical failure, no new invariant
B35|2026-09-12|Grayscale decoder draft guessed profile names + palette fallback and rebuilt GPU palette perupdate|Require exact RGB/gray IDs + fixed integer-luma LUT + devicecache; V60,V63,V68
B36|2026-09-12|Gray ring regression import exceeded configured Ruff layout|Format imports; mechanical failure, no new invariant
B37|2026-09-12|Host cargo binary does not support rustup +stable directive|Use rustup run stable cargo; existing root V22 covers toolchain selection
B38|2026-09-12|Maturin CLI absent; new Python test guessed reset/step instead of existing reset_batch/step_batch|Rebuild via uv sync native; use actual batch API; mechanical harness fixes, no new invariant
B39|2026-09-12|Reward campaign nested imports lacked Ruff separation|Format nested import block; mechanical failure, no new invariant
B40|2026-09-12|GPU preflight hit overlapping hazards; native Death events repeat for one life and reward validator rejected count>1|V73; normalize death presence in native reward helper; preserve failed artifacts and rerun full gate
B41|2026-09-12|New vector boundary used constant `getattr`, rejected by Ruff B009|Use direct typed attributes; mechanical lint failure, no new invariant
B42|2026-09-12|Vector evaluation test imports were not Ruff-sorted|Sort local imports; mechanical lint failure, no new invariant
B43|2026-09-12|Vector epsilon validation exceeded Ruff line limit|Wrap predicate; mechanical lint failure, no new invariant
