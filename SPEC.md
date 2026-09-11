# Native Gymnasium training + color-coded run dashboard

Standalone spec for this project. Native Rust remains sole game implementation.

§G
G1|Expose native Dodge game/data boundary through Gymnasium Env + vector Env.
G2|Train model against native board observation first; keep learner/policy replaceable.
G3|Publish resumable checkpoints, metrics, evaluation, and provenance per run.
G4|Show training runs in simple dashboard with deterministic lifecycle/quality colors.
G5|Measure comparable learning via survival, reward, throughput, and locked holdout; no 1:1 upstream policy copy.

§C
C1|Rust owns state, physics, reward, terminal state, observations, hashes, pixels.
C2|Python wrapper adds ⊥ game semantics.
C3|MVP observation profile `board-v1`; pixel/ML/hazard profiles separate adapters.
C4|Action contract = existing nine actions; default `step_frames=4`.
C5|One-life train/eval default; training-lives out of scope.
C6|Use current frozen 70/30 seed manifest; train/inner selection only training seeds.
C7|Holdout report-only until checkpoint selection frozen.
C8|Artifacts under `history/dodge/gymnasium/<run_id>/`.
C9|Dashboard read-only; no checkpoint mutation, trainer control, remote allocation, or active-run replay.
C10|No LuaJIT/Pemsa runtime, upstream 3,076-float observation, VelocityFlow, or Pygame GUI.
C11|No new game engine, DQN/pixel/hazard algorithm, Colab lifecycle, or HPO work in this slice.

§I
env: `NativeDodgeEnv` → Gymnasium `Env`
vector: `NativeDodgeVectorEnv` → Gymnasium vector semantics over native batch lanes
action: `Discrete(9)` → existing action index/mask order
observation: `Box` → versioned `board-v1`, `float32`, declared shape/layout
reset: `reset(seed, options)` → `(observation, info)`
step: `step(action)` → `(observation, reward, terminated, truncated, info)`
trainer: `GymTrainingRunner` → existing native PPO first; SB3 adapter optional
manifest: `run.json` → immutable engine/config/learner/observation/seed provenance
status: `status.json` → atomic lifecycle snapshot
metrics: `metrics.jsonl` → append-only update/evaluation records
report: `summary.json` + `REPORT.md` → final train/holdout/throughput result
dashboard: existing HTTP/WebSocket dashboard → catalog, status cards, metrics charts, color badges

§L
L1|Model observes native board state.
L2|Model selects one action ∈ nine-action contract.
L3|Native Rust advances fixed four-frame decision interval.
L4|Environment returns next observation, native reward, terminal/truncation, diagnostics.
L5|Learner updates policy/value from transitions; no Python game logic.
L6|Runner repeats across training seeds; checkpoint selection excludes holdout.
L7|Final evaluation measures survival frames, reward, throughput, and train/holdout gap.

§V
V1|Native Rust sole authority for game transition, reward, terminal state, and observation source.
V2|Gymnasium action space exactly matches existing nine-action contract.
V3|Fixed seed/config/action trace deterministic through wrapper and vector boundary.
V4|Each step advances exactly configured `step_frames`; no hidden extra frame.
V5|Native terminal → `terminated=True`; time cap → `truncated=True`; no conflation.
V6|Step after terminal rejects until reset; no implicit auto-reset.
V7|Observation profile shape, dtype, layout, bounds, and version stable inside checkpoint ABI.
V8|Single-lane and vector-lane outputs equal for same seed/config/action schedule.
V9|`info` includes seed, native frame, decision index, survival frames, score, action, config hash, engine schema, state hash.
V10|Run manifest complete and immutable before first learner update.
V11|Checkpoint publication atomic and precedes evaluation.
V12|`status.json` atomic; partial writes never render valid state.
V13|Metrics append-only; dashboard reads bounded; trainer never waits on clients.
V14|State colors deterministic: queued gray, running blue, paused/stale amber, completed green, failed/stopped red.
V15|Gate colors deterministic: pending gray, pass green, warn amber, fail/invalid red.
V16|Every color badge includes text/icon label; color never carries meaning alone.
V17|Completed run with failed gate shows completed state + failed gate; never all-green.
V18|Malformed/mismatched/provenance-incomplete run invalid; invalid run cannot pass.
V19|Dashboard default mode cannot write artifacts or issue controls.
V20|Comparable claims require aligned engine config, cadence, reward, observation, seeds, and metric definition.
V21|No learned/pass label before declared training gate and final evaluation requirements pass.
V22|Native verification uses stable Cargo compatible with workspace edition before build/test.
V23|Python verification follows successful `uv sync --extra native --extra training --group dev`.
V24|Plain uv sync builds native wheel without hidden shell PATH or toolchain state.
V25|Native wheel resolution excludes benchmark-only dependencies from runtime/data surface.

§T
id|status|task|cites
---|---|---|---
T1|x|Create uv project, native workspace, Rust toolchain, and copied cartridge asset|C1,C10,V22-V24
T2|x|Copy `dodge-core`, `dodge-batch`, `dodge-python`; add isolated owned-NumPy data boundary and smoke tests|C1,C2,V1,V7,V25
T3|.|Implement `NativeDodgeEnv` Gymnasium API over one native lane|V2-V9,I.env
T4|.|Implement `NativeDodgeVectorEnv` over `NativeBatchEnvironment`; verify lane/reset equivalence|V3,V4,V8,I.vector
T5|.|Connect existing native PPO through Gymnasium contract; keep policy replaceable|G2,L1-L6,I.trainer
T6|.|Add run manifest, atomic status, append-only metrics, checkpoint, resume, and final report|V10-V13,I.manifest,I.status,I.metrics
T7|.|Add frozen 70/30 training/inner/holdout evaluation and comparable metric summary|C6,C7,V20,V21,I.report
T8|.|Extend dashboard catalog for Gymnasium runs without adding controls|C8,C9,V12-V19,I.dashboard
T9|.|Add color-coded run cards, gate badges, stale/invalid states, and accessible labels|V14-V18
T10|.|Add run detail charts for survival, reward, throughput, train, inner, and holdout|G3,G4,V13,V20
T11|.|Add Gymnasium checker, deterministic trace, vector equivalence, artifact, and dashboard fixture tests|V2-V19
T12|.|Run bounded end-to-end smoke: native env → trainer → checkpoint → dashboard card|V1,V6,V10-V19
T13|.|Run frozen-manifest campaign and publish comparable result; no upstream-equivalence claim|G5,C6,C7,V20,V21

§B
id|date|cause|fix
B1|2026-09-09|System `cargo` was 1.82 while copied workspace required edition-2024 support|Use stable toolchain explicitly; V22
B2|2026-09-09|Python checks ran before project dependencies were synchronized|Run declared uv sync before checks; V23
B3|2026-09-09|First Python verification left five Ruff line-length violations|Run Ruff after source changes and format before acceptance
B4|2026-09-09|Copied test-only `panic!` calls violated workspace Clippy policy|Use assertion-based failure paths in copied tests; keep game logic unchanged
B5|2026-09-09|Copied native data code had three Clippy violations under current toolchain|Keep explicit lint allowance for data helper and remove needless borrows
B6|2026-09-09|uv isolated maturin build ignored shell Cargo override and rejected edition-2024 manifests|Use edition-2021-compatible project metadata; verify plain uv sync
B7|2026-09-09|Copied batch benchmark dev dependency resolved edition-2024 `clap_builder` under system Cargo|Exclude benchmark-only dependency/file from game/data project; V25
B8|2026-09-10|Plain uv build exposed copied let-chain syntax and `cfg(test)` warnings under system Cargo|Rewrite equivalent control flow and declare the build-script cfg; preserve game/data behavior and rerun V24
B9|2026-09-10|`cfg(test)` is checked in each copied crate, so the core build script could not register it for the batch crate|Declare `cfg(test)` at the workspace lint boundary; rerun the isolated wheel build
B10|2026-09-10|Host dynamic-loader path omitted `libz.so.1`, so direct NumPy smoke import failed after installation|Add a project-local `uv` launcher that discovers Nix zlib without changing native/data behavior
