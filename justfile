set shell := ["bash", "-euo", "pipefail", "-c"]

project_root := justfile_directory()
history_root := env_var_or_default("DODGE_HISTORY_ROOT", "history/dodge/gymnasium")

default:
    @just --list

# Install the locked Python environment and native optional dependency.
setup:
    @cd "{{project_root}}" && uv sync --locked --extra native --extra training --group dev

# Run the complete Python and Rust verification set.
verify: test lint native-test native-fmt-check native-lint

# Run Python tests through the host-compatible uv wrapper.
test:
    @cd "{{project_root}}" && scripts/uv-run pytest

# Check Python lint rules.
lint:
    @cd "{{project_root}}" && scripts/uv-run ruff check .

# Exercise the installed native Python boundary.
smoke:
    @cd "{{project_root}}" && scripts/uv-run dodge-native-smoke

# Run all native Rust tests.
native-test:
    @cd "{{project_root}}" && rustup run stable cargo test --manifest-path native/Cargo.toml --workspace --locked

# Check Rust formatting without changing files.
native-fmt-check:
    @cd "{{project_root}}" && rustup run stable cargo fmt --manifest-path native/Cargo.toml --all -- --check

# Apply Rust formatting.
native-fmt:
    @cd "{{project_root}}" && rustup run stable cargo fmt --manifest-path native/Cargo.toml --all

# Run Rust clippy with warnings treated as errors.
native-lint:
    @cd "{{project_root}}" && rustup run stable cargo clippy --manifest-path native/Cargo.toml --workspace --all-targets --all-features -- -D warnings

# Start one bounded image-only CNN/DDQN run.
run run_id="cnn-image-ddqn-baseline-002" steps="256" stack_size="4" device="auto":
    @cd "{{project_root}}" && scripts/uv-run dodge-cnn-image-run --history-root "{{history_root}}" --run-id "{{run_id}}" --steps "{{steps}}" --stack-size "{{stack_size}}" --device "{{device}}"

# Launch the LaunchSpark-parity Pygame dashboard. Watch Agent starts the
# Tailscale-facing browser replay server on demand.
dashboard:
    @cd "{{project_root}}" && exec scripts/uv-run dodge-cnn-image-dashboard

# Start the standalone Tailscale-facing replay server. The dashboard starts
# its own copy on demand when Watch Agent is clicked; use this target only for
# a separate server process.
replay-server:
    @cd "{{project_root}}" && exec scripts/uv-run dodge-cnn-image-replay-server

# Serve one saved checkpoint as a browser replay over Tailscale.
replay checkpoint steps="360" seed="42":
    @cd "{{project_root}}" && exec scripts/uv-run dodge-cnn-image-replay-server --checkpoint "{{checkpoint}}" --steps "{{steps}}" --seed "{{seed}}"

# Render a deterministic dashboard frame without opening a training session.
dashboard-screenshot path="artifacts/dashboard.png":
    @cd "{{project_root}}" && SDL_VIDEODRIVER=dummy scripts/uv-run dodge-cnn-image-dashboard --screenshot "{{path}}"

# Keep the old command name as an explicit local alias.
dashboard-local:
    @just dashboard

# Show the dashboard/replay transport split and the expected Tailscale URL.
dashboard-url:
    @echo "Dashboard: local LaunchSpark-parity Pygame window"
    @echo "Replay: http://100.100.169.122:8890/replay/<token> (created by Watch Agent)"
