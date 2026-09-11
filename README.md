# Dodge Gymnasium

Small standalone project containing the native Rust Dodge game engine, its
Python data exposures, and an isolated CNN/DDQN training variant. The native
engine remains the game authority.

## Setup

```bash
uv sync --extra native --extra training --group dev
```

This uses the same local Python/Rust stack as `rl-learning`, with Gymnasium and
optional Stable-Baselines3/TensorBoard added for the planned training layer.
On this Nix host, use `scripts/uv-run` for Python commands so NumPy can find
the host zlib library; on ordinary systems it simply delegates to `uv run`.

Check the native workspace and Python boundary:

```bash
rustup run stable cargo test --manifest-path native/Cargo.toml --workspace --locked
rustup run stable cargo fmt --manifest-path native/Cargo.toml --all -- --check
rustup run stable cargo clippy --manifest-path native/Cargo.toml --workspace --all-targets --all-features -- -D warnings
scripts/uv-run pytest
scripts/uv-run ruff check .
scripts/uv-run dodge-native-smoke
```

## Copied native surface

- `native/crates/dodge-core`: deterministic Dodge game logic and typed state.
- `native/crates/dodge-batch`: serial/parallel lanes plus board, ML, pixel, and
  hazard data exposures.
- `native/crates/dodge-python`: PyO3/NumPy extension imported as `dodge_native`.
- `src/dodge/game/dodge.p8`: graphics asset required by the native build.
- `src/dodge_native_game/batch.py`: isolated owned-NumPy wrapper around the
  extension; it does not depend on the old repository’s legacy modules.

The native boundary exposes full snapshots, state and pixel hashes, indexed
128×128 pixels, board tensors, 225-value ML observations, player positions,
rewards, terminal flags, and frozen-center hazard observations.

## Basic teaching loop

The first training layer is the isolated CNN image Double-DQN variant in
`variants/cnn-image-ddqn`. It wraps one native lane as a Gymnasium environment
and keeps the native batch engine as the game authority. At each decision:

1. The model receives a collision-system grayscale image, initially as four
   stacked 84×84 frames.
2. It selects one of the existing nine movement actions.
3. Rust advances the game for four frames and returns the next observation,
   native reward, terminal state, and diagnostic info.
4. Double-DQN updates from uint8 replay transitions using the online/target
   CNN pair.
5. The runner writes checkpoints, metrics, evaluation results, and provenance.
6. The LaunchSpark-parity Pygame dashboard reads those artifacts and color-codes
   live learning progress, events, checkpoints, and system readings.
7. `Watch Agent` opens a separate browser replay page. It replays a saved
   checkpoint through a fresh native Rust lane and serves the collision-system
   frames over Tailscale, with play/pause, frame stepping, and a timeline.

The model learns by maximizing survival and task reward over many randomized
seeds, while evaluation remains separate from training. The system compares
survival frames, reward, throughput, and generalization—not checkpoint bytes or
an upstream policy architecture.

## Run the first variant

The root `justfile` wraps setup, test, lint, native checks, runs, and the
dashboard. `just dashboard` opens the LaunchSpark-parity desktop dashboard;
`just dashboard-screenshot` renders a deterministic frame for smoke checks.
The dashboard remains a local Pygame control surface, while `Watch Agent`
starts the separate browser replay service on the host's Tailscale address.

The default run is `just run`; custom run arguments are positional:
`just run <run-id> <steps> <stack-size> <device>`.

Start a bounded baseline and publish its artifacts under the variant namespace:

```bash
scripts/uv-run dodge-cnn-image-run \
  --history-root history/dodge/gymnasium \
  --run-id cnn-image-ddqn-baseline-001 \
  --steps 256 --stack-size 4 --device auto
```

Open the live Pygame dashboard and start a native DDQN session:

```bash
just dashboard
```

Click `Watch Agent` after a checkpoint exists to open the native replay page at
the host's Tailscale address on port `8890`. The Pygame dashboard remains the
control surface; the browser page is read-only replay playback.

Render the copied dashboard without training:

```bash
just dashboard-screenshot artifacts/launchspark-dashboard.png
```

The launcher also accepts `--stack-size 1`, `2`, `4`, or `8`; the ablation
campaign remains a separate task so its runs can be compared explicitly.
