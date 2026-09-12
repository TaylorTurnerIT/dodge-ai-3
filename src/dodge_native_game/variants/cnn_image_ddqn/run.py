"""Bounded image-only Double-DQN runner for the first model variant."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import operator
import os
import threading
import time
import tomllib
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .agent import MAX_GRAD_NORM, DoubleDQNAgent
from .diagnostics import (
    action_balance_ratio,
    best_greedy_episode,
    decide_gate,
    effective_decay_steps,
    eval_seed_list,
    holdout_eval_seeds,
    inner_eval_seeds,
    q_spread_stats,
    reward_mix_stats,
)
from .env import ACTION_COUNT, CNNImageDDQNEnv
from .episode_metrics import (
    EpisodeReturnState,
    paired_reward_deltas,
    record_episode_step,
)
from .model import INITIALIZATION_ID, AtariCnnQNetwork
from .pixels import (
    COLLISION_PROFILE,
    GRAY_PROFILE,
    OBSERVATION_PROFILES,
    RGB_PROFILE,
    observation_shape,
)
from .provenance import git_source_provenance, infer_parent_run_id, sha256_file
from .replay import ReplayBuffer
from .reward_profiles import PROFILES, weighted_components
from .reward_profiles import contract as reward_contract
from .rewards import UNCONTROLLED_SCORE_PER_ENEMY, RewardConfig
from .run_artifacts import VARIANT_ID, RunArtifactWriter
from .seed_pool import training_seed_pool

DEFAULT_HISTORY_ROOT = Path("history/dodge/gymnasium")
DEFAULT_STEPS = 256
DEFAULT_SEED = 42
DEFAULT_STACK_SIZE = 4
DEFAULT_BATCH_SIZE = 32
DEFAULT_WARMUP_STEPS = 32
DEFAULT_UPDATE_EVERY = 4
DEFAULT_TARGET_SYNC_INTERVAL = 100
DEFAULT_LOG_INTERVAL = 16
DEFAULT_EVAL_EPISODES = 4
DEFAULT_EVAL_STEPS = 64
DEFAULT_OBSERVATION_PROFILE = COLLISION_PROFILE
RGB_REPLAY_HEADROOM = 0.20
DIAGNOSTIC_SAMPLE_CADENCE = (
    "first-update-and-next-update-before-log-boundary-or-segment-end"
)

# Native FrameEvent bits; must match event_flags_code in dodge-python.
EVENT_SPAWN = 1 << 0
EVENT_COLLISION = 1 << 1
EVENT_DEATH = 1 << 2
EVENT_PATTERN = 1 << 3
EVENT_TERMINAL = 1 << 4

EnvFactory = Callable[..., CNNImageDDQNEnv]


@dataclass(frozen=True, slots=True)
class TrainingCommand:
    """A dashboard request handled at a safe point in the learner loop."""

    kind: str
    path: Path | None = None
    config: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class TrainingEvent:
    """Result of an asynchronous dashboard request."""

    kind: str
    ok: bool
    message: str
    path: Path | None = None
    update: int = 0


class TrainingControl:
    """Thread-safe control plane for the native DDQN dashboard.

    The learner remains the owner of the native environment and model. The
    dashboard only queues requests; the loop applies them between environment
    steps or while paused. This keeps the hot path deterministic and makes
    Pause, Save, Load, and Game Config work without sharing mutable tensors.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._paused = False
        self._stopped = False
        self._commands: deque[TrainingCommand] = deque()
        self._events: deque[TrainingEvent] = deque()

    @property
    def paused(self) -> bool:
        with self._condition:
            return self._paused

    @property
    def stopped(self) -> bool:
        with self._condition:
            return self._stopped

    def toggle_pause(self) -> bool:
        with self._condition:
            self._paused = not self._paused
            self._condition.notify_all()
            return self._paused

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._paused = False
            self._condition.notify_all()

    def wait(self, timeout: float = 0.05) -> None:
        with self._condition:
            self._condition.wait(timeout=timeout)

    def request_save(self, path: Path, *, kind: str = "save") -> None:
        with self._condition:
            self._commands.append(TrainingCommand(kind=kind, path=path))
            self._condition.notify_all()

    def request_load(self, path: Path) -> None:
        with self._condition:
            self._commands.append(TrainingCommand(kind="load", path=path))
            self._condition.notify_all()

    def request_config(self, config: Mapping[str, Any]) -> None:
        with self._condition:
            self._commands.append(TrainingCommand(kind="config", config=dict(config)))
            self._condition.notify_all()

    def drain_commands(self) -> list[TrainingCommand]:
        with self._condition:
            commands = list(self._commands)
            self._commands.clear()
            return commands

    def emit(self, event: TrainingEvent) -> None:
        with self._condition:
            self._events.append(event)

    def drain_events(self) -> list[TrainingEvent]:
        with self._condition:
            events = list(self._events)
            self._events.clear()
            return events


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_variant_config() -> dict[str, Any]:
    path = _project_root() / "variants" / "cnn-image-ddqn" / "config.toml"
    try:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    except (FileNotFoundError, OSError, tomllib.TOMLDecodeError):
        return {
            "variant_id": VARIANT_ID,
            "observation_id": "collision-image-v1",
            "game": {"step_frames": 4, "action_count": ACTION_COUNT},
            "model": {
                "dueling": True,
                "hidden_size": 512,
                "gamma": 0.99,
                "learning_rate": 0.00025,
                "gradient_clip_norm": 10.0,
            },
            "replay": {"capacity": 100_000},
            "target": {"sync_interval": DEFAULT_TARGET_SYNC_INTERVAL},
            "exploration": {
                "epsilon_start": 1.0,
                "epsilon_final": 0.1,
                "epsilon_decay_steps": 1_000_000,
            },
        }


def _resolve_observation_profile(
    requested: str | None, config: Mapping[str, Any]
) -> str:
    """Resolve and validate the immutable observation profile for a run."""

    observation_config = config.get("observation")
    configured: object = config.get(
        "observation_profile",
        config.get("observation_id", DEFAULT_OBSERVATION_PROFILE),
    )
    if isinstance(observation_config, Mapping):
        configured = observation_config.get(
            "profile", observation_config.get("id", configured)
        )
    profile = configured if requested is None else requested
    if profile not in OBSERVATION_PROFILES:
        choices = ", ".join(OBSERVATION_PROFILES)
        raise ValueError(f"unknown observation profile {profile!r}; choose {choices}")
    return str(profile)


def _profile_for_shape(shape: tuple[int, int, int]) -> str:
    """Infer a legacy-compatible profile only when shape makes it unambiguous."""

    if len(shape) == 3 and tuple(shape[1:]) == (84, 84) and shape[0] >= 1:
        return COLLISION_PROFILE
    if (
        len(shape) == 3
        and tuple(shape[1:]) == (128, 128)
        and shape[0] >= 3
        and shape[0] % 3 == 0
    ):
        return RGB_PROFILE
    if len(shape) == 3 and tuple(shape[1:]) == (128, 128) and shape[0] >= 1:
        return GRAY_PROFILE
    raise ValueError(f"unsupported observation shape: {shape}")


def _checkpoint_profile(payload: Mapping[str, object]) -> str:
    """Read a checkpoint profile, treating absent legacy metadata as collision."""

    profile = payload.get("observation_profile", COLLISION_PROFILE)
    if profile not in OBSERVATION_PROFILES:
        choices = ", ".join(OBSERVATION_PROFILES)
        raise ValueError(
            f"checkpoint observation profile {profile!r} is invalid; choose {choices}"
        )
    return str(profile)


def _checkpoint_shape(payload: Mapping[str, object]) -> tuple[int, int, int]:
    raw_shape = payload.get("observation_shape", ())
    try:
        shape = tuple(int(item) for item in raw_shape)  # type: ignore[union-attr]
    except (TypeError, ValueError) as error:
        raise ValueError(
            "checkpoint observation_shape must be a 3-item sequence"
        ) from error
    if len(shape) != 3:
        raise ValueError("checkpoint observation_shape must be a 3-item sequence")
    return shape


def _checkpoint_stack_size(payload: Mapping[str, object], profile: str) -> int:
    """Read stack depth, deriving it for old collision checkpoints."""

    raw_stack = payload.get("stack_size")
    if raw_stack is None:
        shape = _checkpoint_shape(payload)
        if profile == COLLISION_PROFILE:
            raw_stack = shape[0]
        else:
            raise ValueError("native RGB checkpoint is missing stack_size")
    if isinstance(raw_stack, bool):
        raise ValueError("checkpoint stack_size must be a positive integer")
    try:
        stack_size = operator.index(raw_stack)
    except TypeError as error:
        raise ValueError("checkpoint stack_size must be a positive integer") from error
    if stack_size < 1:
        raise ValueError("checkpoint stack_size must be a positive integer")
    return int(stack_size)


def _validate_checkpoint_observation(
    payload: Mapping[str, object],
    *,
    observation_profile: str,
    stack_size: int,
    expected_shape: tuple[int, int, int],
) -> None:
    """Reject profile/depth/shape mismatches before loading network weights."""

    checkpoint_profile = _checkpoint_profile(payload)
    if checkpoint_profile != observation_profile:
        raise ValueError(
            "checkpoint observation profile does not match the current session: "
            f"{checkpoint_profile!r} != {observation_profile!r}"
        )
    checkpoint_stack = _checkpoint_stack_size(payload, checkpoint_profile)
    if checkpoint_stack != stack_size:
        raise ValueError(
            "checkpoint stack_size does not match the current session: "
            f"{checkpoint_stack} != {stack_size}"
        )
    checkpoint_shape = _checkpoint_shape(payload)
    if checkpoint_shape != expected_shape:
        raise ValueError(
            "checkpoint observation shape does not match the current session: "
            f"{checkpoint_shape} != {expected_shape}"
        )


def _available_host_memory_bytes() -> int:
    """Return currently available host memory for the one-time replay gate."""

    try:
        import psutil

        available = int(psutil.virtual_memory().available)
    except ImportError:
        try:
            available = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (AttributeError, OSError, ValueError) as error:
            raise RuntimeError("cannot measure available host memory") from error
    if available < 1:
        raise RuntimeError("available host memory measurement was invalid")
    return available


def _legacy_replay_storage_bytes(capacity: int, shape: tuple[int, int, int]) -> int:
    """Estimate explicit uint8 state + next-state replay allocation."""

    image_bytes = int(np.prod(shape, dtype=np.int64)) * int(capacity) * 2
    scalar_bytes = int(capacity) * (
        np.dtype(np.int64).itemsize
        + np.dtype(np.float32).itemsize
        + np.dtype(np.bool_).itemsize
    )
    return image_bytes + scalar_bytes


def _pixel_replay_class() -> type[Any]:
    """Import the packed native RGB replay lazily so collision runs stay stable."""

    from .pixel_replay import NativePixelReplayBuffer

    return NativePixelReplayBuffer


def _pixel_replay_storage_bytes(
    replay_class: type[Any], capacity: int, stack_size: int
) -> int:
    estimator = getattr(replay_class, "estimated_storage_bytes", None)
    if callable(estimator):
        return int(estimator(capacity, stack_size))
    from .pixel_replay import estimated_storage_bytes

    return int(estimated_storage_bytes(capacity, stack_size))


def _replay_storage_plan(
    *,
    observation_profile: str,
    capacity: int,
    stack_size: int,
    observation_shape_value: tuple[int, int, int],
) -> tuple[type[Any] | None, dict[str, Any]]:
    """Plan replay storage and gate RGB allocation before constructing arrays."""

    if observation_profile == COLLISION_PROFILE:
        return None, {
            "encoding": "uint8-state-next-v1",
            "capacity": capacity,
            "estimated_bytes": _legacy_replay_storage_bytes(
                capacity, observation_shape_value
            ),
            "available_bytes": None,
            "headroom_fraction": None,
        }

    replay_class = _pixel_replay_class()
    estimated_bytes = _pixel_replay_storage_bytes(replay_class, capacity, stack_size)
    available_bytes = _available_host_memory_bytes()
    required_bytes = math.ceil(estimated_bytes / (1.0 - RGB_REPLAY_HEADROOM))
    if required_bytes > available_bytes:
        raise MemoryError(
            "native RGB replay requires "
            f"{estimated_bytes:,} bytes plus {RGB_REPLAY_HEADROOM:.0%} host-memory "
            f"headroom, but only {available_bytes:,} bytes are available"
        )
    return replay_class, {
        "encoding": "native-palette4-frame-ring-v1",
        "capacity": capacity,
        "estimated_bytes": estimated_bytes,
        "available_bytes": available_bytes,
        "required_bytes_with_headroom": required_bytes,
        "headroom_fraction": RGB_REPLAY_HEADROOM,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _diagnostics_requested(
    *,
    step: int,
    update_every: int,
    log_interval: int,
    steps: int,
    optimizer_step: int,
    optimizer_step_start: int,
) -> bool:
    """Sample one optimizer diagnostic at each useful logging boundary."""

    if optimizer_step == optimizer_step_start:
        return True
    # The update at ``step`` is the last one before the next logging boundary
    # when its next scheduled update crosses that boundary. This handles
    # update/log intervals that are not integer multiples of one another.
    before_log_boundary = (step - 1) // log_interval != (
        step + update_every - 1
    ) // log_interval
    return before_log_boundary or step + update_every > steps


def _observation_to_uint8(
    observation: object, expected_shape: tuple[int, int, int]
) -> np.ndarray:
    raw = np.asarray(observation)
    if raw.shape != expected_shape:
        raise ValueError(
            f"observation must have shape {expected_shape}, got {raw.shape}"
        )
    if raw.dtype == np.uint8:
        return np.ascontiguousarray(raw, dtype=np.uint8).copy()
    value = np.asarray(raw, dtype=np.float32)
    if value.shape != expected_shape:
        raise ValueError(
            f"observation must have shape {expected_shape}, got {value.shape}"
        )
    if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError("observation values must be finite and in [0, 1]")
    return np.rint(value * np.float32(255.0)).astype(np.uint8, copy=True)


def epsilon_at(
    step: int,
    *,
    start: float,
    final: float,
    decay_steps: int,
) -> float:
    """Return linearly annealed epsilon at a one-based training step."""

    if decay_steps < 1:
        raise ValueError("decay_steps must be positive")
    fraction = min(max(float(step), 0.0) / float(decay_steps), 1.0)
    return float(start + fraction * (final - start))


def _choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return requested


def _configure_torch_backend(device: str) -> None:
    """Avoid probing unavailable CPU kernels on this host.

    Some PyTorch builds ship NNPACK but advertise it before checking the
    machine's instruction set. On unsupported CPUs that path repeatedly emits
    initialization warnings and can terminate a forward pass with SIGILL.
    Keep CUDA behavior unchanged; native CPU runs use PyTorch's other kernels.
    """

    if device != "cpu":
        return
    nnpack = getattr(torch.backends, "nnpack", None)
    set_flags = getattr(nnpack, "set_flags", None)
    if callable(set_flags):
        set_flags(False)


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _checkpoint_payload(
    agent: DoubleDQNAgent,
    *,
    step: int,
    seed: int,
    observation_profile: str = DEFAULT_OBSERVATION_PROFILE,
    stack_size: int | None = None,
    run_id: str | None = None,
    n_step: int = 1,
    reward_profile: str = "survival-v1",
    target_sync_count: int = 0,
    source: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    profile = _resolve_observation_profile(observation_profile, {})
    agent_shape = tuple(int(value) for value in agent.observation_shape)
    if stack_size is None:
        if profile == RGB_PROFILE:
            if agent_shape[0] % 3:
                raise ValueError(
                    "native RGB checkpoint requires a channel count divisible by 3"
                )
            stack_size = agent_shape[0] // 3
        else:
            stack_size = agent_shape[0]
    expected_shape = observation_shape(profile, int(stack_size))
    if agent_shape != expected_shape:
        raise ValueError(
            "checkpoint observation profile/shape mismatch: "
            f"{profile!r} with stack {stack_size} expects {expected_shape}, "
            f"got {agent_shape}"
        )
    return {
        "checkpoint_schema_version": 2,
        "native_reward_contract": reward_contract(reward_profile),
        "variant_id": VARIANT_ID,
        "step": step,
        "global_environment_step": step,
        "seed": seed,
        "n_step": n_step,
        "run_id": run_id,
        "initialization_id": INITIALIZATION_ID,
        "observation_profile": profile,
        "stack_size": int(stack_size),
        "model_input_size": int(expected_shape[1]),
        "observation_shape": list(expected_shape),
        "num_actions": agent.num_actions,
        "online_network": agent.online_network.state_dict(),
        "target_network": agent.target_network.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "optimizer_steps": agent.optimizer_steps,
        "target_sync_count": int(target_sync_count),
        "source": dict(source or {}),
    }


def _epsilon_entropy(epsilon: float, num_actions: int) -> float:
    """Entropy of the epsilon-greedy action distribution.

    DDQN has no policy softmax. This is the exact exploration-distribution
    entropy, which gives LaunchSpark's Policy Entropy panel an honest native
    DDQN signal instead of pretending Q-values are probabilities.
    """

    if num_actions < 1:
        raise ValueError("num_actions must be positive")
    random_probability = float(epsilon) / num_actions
    greedy_probability = 1.0 - float(epsilon) + random_probability
    terms = [greedy_probability]
    terms.extend([random_probability] * (num_actions - 1))
    return float(-sum(value * np.log(value) for value in terms if value > 0.0))


def _load_checkpoint(
    agent: DoubleDQNAgent,
    path: Path,
    *,
    observation_profile: str | None = None,
    stack_size: int | None = None,
    expected_n_step: int | None = None,
) -> dict[str, Any]:
    """Load one native DDQN checkpoint into an existing learner."""

    payload = torch.load(path, map_location=agent.device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    if payload.get("variant_id") != VARIANT_ID:
        raise ValueError("checkpoint variant does not match cnn-image-ddqn")
    if expected_n_step is not None and payload.get("n_step", 1) != expected_n_step:
        raise ValueError("checkpoint n_step does not match training objective")
    agent_shape = tuple(int(value) for value in agent.observation_shape)
    expected_profile = (
        _profile_for_shape(agent_shape)
        if observation_profile is None
        else _resolve_observation_profile(observation_profile, {})
    )
    expected_stack = (
        agent_shape[0] // (3 if expected_profile == RGB_PROFILE else 1)
        if stack_size is None
        else int(stack_size)
    )
    _validate_checkpoint_observation(
        payload,
        observation_profile=expected_profile,
        stack_size=expected_stack,
        expected_shape=agent_shape,
    )
    if int(payload.get("num_actions", -1)) != agent.num_actions:
        raise ValueError("checkpoint action count does not match the current session")
    agent.online_network.load_state_dict(payload["online_network"])
    target = payload.get("target_network")
    if target is None:
        agent.sync_target()
    else:
        agent.target_network.load_state_dict(target)
        agent.target_network.eval()
    optimizer = payload.get("optimizer")
    if optimizer is not None:
        agent.optimizer.load_state_dict(optimizer)
    agent.optimizer_steps = _checkpoint_counter(payload, "optimizer_steps")
    return payload


def _checkpoint_counter(
    payload: Mapping[str, object], key: str, *, default: int = 0
) -> int:
    """Read a non-negative integral checkpoint counter without truncation."""

    raw = payload.get(key, default)
    if isinstance(raw, bool):
        raise ValueError(f"checkpoint {key} must be a non-negative integer")
    try:
        value = operator.index(raw)
    except TypeError as error:
        raise ValueError(f"checkpoint {key} must be a non-negative integer") from error
    if value < 0:
        raise ValueError(f"checkpoint {key} must be a non-negative integer")
    return int(value)


def _checkpoint_global_step(payload: Mapping[str, object]) -> int:
    """Read global progress while accepting legacy step-only checkpoints."""

    legacy_step = _checkpoint_counter(payload, "step")
    if "global_environment_step" not in payload:
        return legacy_step
    global_step = _checkpoint_counter(payload, "global_environment_step")
    if "step" in payload and global_step != legacy_step:
        raise ValueError("checkpoint global_environment_step conflicts with step")
    return global_step


def _native_game_config(
    config: Mapping[str, Any] | None,
    *,
    default_step_frames: int,
) -> dict[str, Any]:
    """Reduce the LaunchSpark game controls to native engine settings.

    The Rust engine currently owns difficulty, pattern, and powerup switches.
    The remaining LaunchSpark controls stay in the dashboard/session metadata
    for parity and future native exposure; they do not invent Python-side game
    semantics.
    """

    values = {
        "difficulty": 2,
        "patterns": True,
        "powerups": True,
        "normal_enemies": True,
        "kamikaze_enemies": True,
        "spawn_rate_percent": 100,
        "action_repeat": default_step_frames,
        "lives": 0,
    }
    if config:
        values.update(config)
    action_repeat = int(values["action_repeat"])
    # Native BatchConfig deliberately accepts decision intervals 3..5. Keep
    # the upstream slider range, but clamp its value at this boundary.
    native_step_frames = min(5, max(3, action_repeat))
    return {
        **values,
        "difficulty": int(values["difficulty"]),
        "patterns": bool(values["patterns"]),
        "powerups": bool(values["powerups"]),
        "native_step_frames": native_step_frames,
    }


def _checkpoint_path(
    writer: RunArtifactWriter, requested: Path
) -> tuple[Path, str | None]:
    """Resolve a dashboard checkpoint and return its safe run-relative name."""

    path = Path(requested)
    if not path.is_absolute():
        path = writer.paths.root / path
    path = path.expanduser()
    run_root = writer.paths.root.resolve()
    try:
        relative = path.resolve().relative_to(run_root)
    except ValueError:
        return path, None
    return path, relative.as_posix()


def _evaluate(
    agent: DoubleDQNAgent,
    env: CNNImageDDQNEnv,
    *,
    seed: int,
    episodes: int,
    max_steps: int,
    seed_offset: int = 10_000,
) -> dict[str, Any]:
    rewards: list[float] = []
    survival_frames: list[int] = []
    seeds: list[int] = eval_seed_list(seed, episodes, offset=seed_offset)
    q_samples: list[list[float]] = []
    dead_units: list[float] = []
    action_counts = [0] * agent.num_actions
    terminated_rows: list[bool] = []
    for eval_seed in seeds:
        observation, _ = env.reset(seed=eval_seed)
        episode_reward = 0.0
        frames = 0
        episode_terminated = False
        for _ in range(max_steps):
            try:
                q_values, dead_fraction = agent.evaluation_values(observation)
                q_samples.append([float(value) for value in q_values.reshape(-1)])
                dead_units.append(float(dead_fraction))
                action = int(q_values.reshape(-1).argmax())
            except (RuntimeError, ValueError):
                action = int(agent.select_action(observation, 0.0))
            action_counts[action] += 1
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            frames += int(info["native_frames_advanced"])
            if terminated or truncated:
                episode_terminated = bool(terminated)
                break
        rewards.append(episode_reward)
        survival_frames.append(frames)
        terminated_rows.append(episode_terminated)
    spread = (
        q_spread_stats(q_samples)
        if q_samples
        else {"mean": 0.0, "std": 0.0, "gap": 0.0}
    )
    return {
        "episodes": episodes,
        "observation_profile": getattr(
            env, "observation_profile", DEFAULT_OBSERVATION_PROFILE
        ),
        "max_steps_per_episode": max_steps,
        "seed_offset": seed_offset,
        "seeds": seeds,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_survival_frames": (
            float(np.mean(survival_frames)) if survival_frames else 0.0
        ),
        "rewards": rewards,
        "survival_frames": survival_frames,
        "q_mean": float(spread["mean"]),
        "q_std": float(spread["std"]),
        "q_gap": float(spread["gap"]),
        "dead_units_mean": float(np.mean(dead_units)) if dead_units else 0.0,
        "action_counts": action_counts,
        "action_balance": action_balance_ratio(action_counts),
        "terminated": terminated_rows,
        "censored": [not value for value in terminated_rows],
        "censored_share": (
            float(np.mean([not value for value in terminated_rows]))
            if terminated_rows
            else 0.0
        ),
    }


def _counterfactual_eval(
    agent: DoubleDQNAgent,
    env: CNNImageDDQNEnv,
    *,
    seed: int,
    episodes: int,
    max_steps: int,
    forced_action: int,
    seed_offset: int = 30_000,
) -> dict[str, Any]:
    """Greedy rollout with the first move forced to a rarely used action.

    Answers the article's forced-move question: if the policy avoids an
    action, does forcing it once lead somewhere better or worse? Uses fresh
    resets on the same env instance; never touches training state.
    """

    seeds: list[int] = eval_seed_list(seed, episodes, offset=seed_offset)
    baseline_rewards: list[float] = []
    forced_rewards: list[float] = []
    for eval_seed in seeds:
        rewards_for_seed: list[float] = []
        for force_first_action in (False, True):
            observation, _ = env.reset(seed=eval_seed)
            episode_reward = 0.0
            try:
                for decision in range(max_steps):
                    action = (
                        int(forced_action)
                        if force_first_action and decision == 0
                        else int(agent.select_action(observation, 0.0))
                    )
                    observation, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += float(reward)
                    if terminated or truncated:
                        break
            except (RuntimeError, ValueError):
                pass
            rewards_for_seed.append(episode_reward)
        baseline_rewards.append(rewards_for_seed[0])
        forced_rewards.append(rewards_for_seed[1])
    paired = paired_reward_deltas(
        zip(seeds, baseline_rewards, strict=True),
        zip(seeds, forced_rewards, strict=True),
    )
    return {
        "episodes": episodes,
        "observation_profile": getattr(
            env, "observation_profile", DEFAULT_OBSERVATION_PROFILE
        ),
        "forced_action": int(forced_action),
        "seed_offset": seed_offset,
        "seeds": seeds,
        "mean_reward": float(np.mean(forced_rewards)) if forced_rewards else 0.0,
        "rewards": forced_rewards,
        "baseline_rewards": baseline_rewards,
        "reward_deltas": [row["delta"] for row in paired["rows"]],
        "mean_reward_delta": paired["summary"]["mean_delta"],
        "paired": paired,
    }


def train_run(
    *,
    history_root: os.PathLike[str] | str = DEFAULT_HISTORY_ROOT,
    run_id: str,
    steps: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
    stack_size: int = DEFAULT_STACK_SIZE,
    observation_profile: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    replay_capacity: int | None = None,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    update_every: int = DEFAULT_UPDATE_EVERY,
    target_sync_interval: int = DEFAULT_TARGET_SYNC_INTERVAL,
    log_interval: int = DEFAULT_LOG_INTERVAL,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    eval_steps: int = DEFAULT_EVAL_STEPS,
    device: str = "auto",
    env_factory: EnvFactory = CNNImageDDQNEnv,
    control: TrainingControl | None = None,
    game_config: Mapping[str, Any] | None = None,
    resume_from: os.PathLike[str] | str | None = None,
    dueling: bool | None = None,
    learning_rate: float | None = None,
    epsilon_decay_steps: int | None = None,
    training_seed_count: int | None = None,
    evaluation_seed: int | None = None,
    checkpoint_steps: Sequence[int] = (),
    n_step: int = 1,
    shaping: bool = False,
    reward_profile: str = "survival-v1",
    w_survival: float | None = None,
    w_death: float | None = None,
    w_score: float | None = None,
    w_uncontrolled: float | None = None,
) -> Path:
    """Run one bounded experiment and return its artifact directory."""

    for name, value in (
        ("steps", steps),
        ("stack_size", stack_size),
        ("batch_size", batch_size),
        ("warmup_steps", warmup_steps),
        ("update_every", update_every),
        ("target_sync_interval", target_sync_interval),
        ("log_interval", log_interval),
        ("eval_episodes", eval_episodes),
        ("eval_steps", eval_steps),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if not 0 <= seed <= 32_767:
        raise ValueError("seed must be between 0 and 32767")
    eval_base_seed = seed if evaluation_seed is None else evaluation_seed
    if training_seed_count is not None or evaluation_seed is not None:
        if not 0 <= eval_base_seed <= 2767:
            raise ValueError(
                "evaluation_seed must be in [0, 2767] to preserve disjoint ranges"
            )
        if eval_episodes > 2768 - eval_base_seed:
            raise ValueError("evaluation seed range would clamp or overlap")
    pool = (
        training_seed_pool(training_seed_count)
        if training_seed_count is not None
        else None
    )
    seed_steps: dict[int, int] = {}
    snapshot_steps = set(checkpoint_steps)
    if any(
        not isinstance(value, int) or not 1 <= value <= steps
        for value in snapshot_steps
    ):
        raise ValueError("checkpoint_steps must fall within the training segment")

    def game_seed(episode_index: int) -> int:
        if pool is not None:
            return pool[episode_index % len(pool)]
        return min(32767, seed + episode_index)

    config = _load_variant_config()
    effective_profile = _resolve_observation_profile(observation_profile, config)
    if isinstance(n_step, bool) or not isinstance(n_step, int) or n_step not in (1, 3):
        raise ValueError("n_step must be 1 or 3")
    if n_step != 1 and effective_profile != COLLISION_PROFILE:
        raise ValueError("n_step > 1 currently requires collision-image-v1")
    if n_step != 1 and control is not None:
        raise ValueError("n_step > 1 currently requires an uncontrolled bounded run")
    engine_config = dict(config.get("game", {}))
    model_config = dict(config.get("model", {}))
    replay_config = dict(config.get("replay", {}))
    target_config = dict(config.get("target", {}))
    exploration_config = dict(config.get("exploration", {}))
    configured_capacity = replay_config.get("capacity", 100_000)
    selected_capacity = (
        configured_capacity if replay_capacity is None else replay_capacity
    )
    if isinstance(selected_capacity, bool):
        raise ValueError("replay_capacity must be positive")
    try:
        selected_capacity = int(selected_capacity)
    except (TypeError, ValueError) as error:
        raise ValueError("replay_capacity must be positive") from error
    if selected_capacity < 1:
        raise ValueError("replay_capacity must be positive")
    step_frames = int(engine_config.get("step_frames", 4))
    num_actions = int(engine_config.get("action_count", ACTION_COUNT))
    selected_game = _native_game_config(
        game_config if game_config is not None else None,
        default_step_frames=step_frames,
    )
    step_frames = int(selected_game["native_step_frames"])
    chosen_device = _choose_device(device)
    _configure_torch_backend(chosen_device)
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    observation_shape_value = observation_shape(effective_profile, stack_size)
    model_input_size = int(observation_shape_value[1])
    replay_class, replay_storage = _replay_storage_plan(
        observation_profile=effective_profile,
        capacity=selected_capacity,
        stack_size=stack_size,
        observation_shape_value=observation_shape_value,
    )
    if n_step > 1:
        replay_storage["estimated_bytes"] += selected_capacity * 4
        replay_storage["bootstrap_discount_dtype"] = "float32"
    if dueling is None:
        dueling_effective = bool(model_config.get("dueling", True))
    else:
        dueling_effective = bool(dueling)
    if learning_rate is None:
        lr_effective = float(model_config.get("learning_rate", 0.00025))
    else:
        lr_effective = float(learning_rate)
    if not lr_effective > 0.0:
        raise ValueError("learning_rate must be positive")
    if epsilon_decay_steps is not None and epsilon_decay_steps < 1:
        raise ValueError("epsilon_decay_steps must be positive")
    reward_defaults = RewardConfig()
    native_reward_contract = reward_contract(reward_profile)
    if reward_profile != "survival-v1" and (shaping or control is not None):
        raise ValueError(
            "native reward profiles reject legacy shaping and live controls"
        )
    component_totals = np.zeros(6, dtype=np.float64)

    def _weight(flag: float | None, default: float) -> float:
        return float(flag) if flag is not None else float(default)

    shaping_weights = {
        "survival_per_frame": _weight(w_survival, reward_defaults.survival_per_frame),
        "death_penalty": _weight(w_death, reward_defaults.death_penalty),
        "score_weight": _weight(w_score, reward_defaults.score_weight),
        "uncontrolled_score_weight": _weight(
            w_uncontrolled, reward_defaults.uncontrolled_score_weight
        ),
    }
    shaping_enabled = bool(shaping)
    source_provenance = git_source_provenance(_project_root())

    manifest = {
        "native_reward_contract": native_reward_contract,
        "schema_version": 1,
        "variant_id": VARIANT_ID,
        "run_id": run_id,
        "seed": seed,
        "observation_profile": effective_profile,
        "n_step": n_step,
        "observation": {
            "profile": effective_profile,
            "shape": list(observation_shape_value),
            "stack_size": stack_size,
            "model_input_size": model_input_size,
        },
        "engine": {
            "observation_id": effective_profile,
            "observation_profile": effective_profile,
            "collision_image_config": (
                "world128-max-composite-u8-hw-v1"
                if effective_profile == COLLISION_PROFILE
                else None
            ),
            "collision_image_version": (
                1 if effective_profile == COLLISION_PROFILE else None
            ),
        },
        "replay": dict(replay_storage),
        "trainer": "native-cnn-image-ddqn",
        "device": chosen_device,
        "initialization_id": INITIALIZATION_ID,
        "source": source_provenance,
        "training_seed_protocol": {
            "mode": (
                "nested-shuffled-cycle-v1"
                if pool is not None
                else "legacy-increment-clamp"
            ),
            "learner_seed": seed,
            "pool_seed": 1729 if pool is not None else None,
            "pool_size": len(pool) if pool is not None else None,
            "seeds": list(pool) if pool is not None else None,
            "evaluation_seed": eval_base_seed,
            "checkpoint_steps": sorted(snapshot_steps),
            "reset_boundary": "natural-episode-end",
        },
    }
    resumed_step = 0
    resume_payload: dict[str, Any] | None = None
    if resume_from is not None:
        loaded_resume_payload = torch.load(
            Path(resume_from), map_location="cpu", weights_only=False
        )
        if not isinstance(loaded_resume_payload, dict):
            raise ValueError("checkpoint must contain a mapping")
        _validate_checkpoint_observation(
            loaded_resume_payload,
            observation_profile=effective_profile,
            stack_size=stack_size,
            expected_shape=observation_shape_value,
        )
        resume_payload = loaded_resume_payload
        if (
            resume_payload.get("native_reward_contract", reward_contract("survival-v1"))
            != native_reward_contract
        ):
            raise ValueError("checkpoint native reward contract does not match")
        if int(resume_payload.get("n_step", 1)) != n_step:
            raise ValueError("checkpoint n_step does not match training objective")
        resumed_step = _checkpoint_global_step(resume_payload)
        manifest["resumed_from"] = str(resume_from)
        manifest["resumed_step"] = resumed_step
        manifest["resume_mode"] = "optimizer-state-with-fresh-replay-rng-env"
        manifest["parent_checkpoint"] = {
            "path": str(resume_from),
            "sha256": sha256_file(Path(resume_from)),
            "run_id": infer_parent_run_id(
                Path(resume_from), resume_payload.get("run_id")
            ),
            "global_environment_step": resumed_step,
            "optimizer_step": _checkpoint_counter(resume_payload, "optimizer_steps"),
            "initialization_id": resume_payload.get("initialization_id"),
            "observation_profile": _checkpoint_profile(resume_payload),
            "stack_size": _checkpoint_stack_size(
                resume_payload, _checkpoint_profile(resume_payload)
            ),
        }
        manifest["resume_state"] = {
            "restored": ["online_network", "target_network", "optimizer"],
            "reset": [
                "replay_buffer",
                "replay_rng",
                "action_rng",
                "numpy_global_rng",
                "torch_rng",
                "environment",
                "episode_accumulator",
                "diagnostic_windows",
                "segment_counters",
            ],
        }
    decay_configured = int(
        epsilon_decay_steps
        if epsilon_decay_steps is not None
        else exploration_config.get("epsilon_decay_steps", 1_000_000)
    )
    decay_effective = effective_decay_steps(decay_configured, steps)
    global_step_start = resumed_step
    global_step_end = resumed_step + steps
    inner_seeds = inner_eval_seeds(eval_base_seed, eval_episodes)
    holdout_seeds = holdout_eval_seeds(eval_base_seed, eval_episodes)
    manifest["eval"] = {
        "protocol": "frozen-train-holdout-v1",
        "observation_profile": effective_profile,
        "observation_shape": list(observation_shape_value),
        "stack_size": stack_size,
        "inner_offset": 10_000,
        "holdout_offset": 20_000,
        "counterfactual_offset": 30_000,
        "counterfactual_protocol": "paired-first-action-v1",
        "inner_seeds_preview": inner_seeds,
        "holdout_seeds_preview": holdout_seeds,
    }
    run_config: dict[str, Any] = {
        "variant_id": VARIANT_ID,
        "game": {"step_frames": step_frames, "action_count": num_actions},
        "observation": {
            "id": effective_profile,
            "profile": effective_profile,
            "shape": list(observation_shape_value),
            "stack_size": stack_size,
            "frame_shape": [3, 128, 128]
            if effective_profile == RGB_PROFILE
            else ([1, 128, 128] if effective_profile == GRAY_PROFILE else [1, 84, 84]),
            "storage_dtype": "uint8",
            "model_dtype": "float32",
        },
        "model": {
            **model_config,
            "architecture": (
                "atari-cnn-dueling-ddqn"
                if dueling_effective
                else "atari-cnn-plain-ddqn"
            ),
            "dueling": dueling_effective,
            "learning_rate": lr_effective,
            "input_channels": observation_shape_value[0],
            "input_size": model_input_size,
            "initialization_id": INITIALIZATION_ID,
            "n_step": n_step,
        },
        "replay": {
            **replay_config,
            **replay_storage,
            "capacity": selected_capacity,
            "batch_size": batch_size,
            "warmup_steps": warmup_steps,
        },
        "target": {
            **target_config,
            "sync_interval": target_sync_interval,
        },
        "exploration": {
            **exploration_config,
            "epsilon_decay_steps": decay_configured,
            "epsilon_decay_steps_configured": decay_configured,
            "epsilon_decay_steps_effective": decay_effective,
            "schedule_unit": "global-environment-steps",
        },
        "evaluation": {
            "protocol": "frozen-train-holdout-v1",
            "observation_profile": effective_profile,
            "observation_shape": list(observation_shape_value),
            "stack_size": stack_size,
            "inner_offset": 10_000,
            "holdout_offset": 20_000,
            "counterfactual_offset": 30_000,
            "counterfactual_protocol": "paired-first-action-v1",
            "episodes": eval_episodes,
            "max_steps_per_episode": eval_steps,
        },
        "run": {
            "steps": steps,
            "segment_steps": steps,
            "global_step_start": global_step_start,
            "global_step_end": global_step_end,
            "seed": seed,
            "training_seed_count": training_seed_count,
            "evaluation_seed": eval_base_seed,
            "device": chosen_device,
            "update_every": update_every,
            "log_interval": log_interval,
            "evaluation_episodes": eval_episodes,
            "evaluation_steps": eval_steps,
        },
        "reward_shaping": {
            "enabled": shaping_enabled,
            **shaping_weights,
        },
        "native_reward_contract": native_reward_contract,
        "dashboard_game_controls": {
            key: selected_game[key]
            for key in (
                "difficulty",
                "patterns",
                "powerups",
                "normal_enemies",
                "kamikaze_enemies",
                "spawn_rate_percent",
                "action_repeat",
                "lives",
            )
        },
    }
    if resume_from is not None:
        run_config["run"]["resumed_from"] = str(resume_from)
        run_config["run"]["resumed_step"] = resumed_step
        run_config["run"]["resume_mode"] = "optimizer-state-with-fresh-replay-rng-env"
    writer = RunArtifactWriter.create(
        history_root,
        run_id,
        manifest=manifest,
        config=run_config,
    )
    if resume_from is not None:
        start_message = f"resuming from {resume_from} (step {resumed_step})"
    else:
        start_message = "training started"
    writer.update_status("running", step=0, episode=0, message=start_message)

    env_kwargs = {
        "stack_size": stack_size,
        "step_frames": step_frames,
        "difficulty": int(selected_game["difficulty"]),
        "patterns": bool(selected_game["patterns"]),
        "powerups": bool(selected_game["powerups"]),
        "observation_profile": effective_profile,
    }
    try:
        try:
            env = env_factory(**env_kwargs)
        except TypeError:
            # Preserve minimal legacy factory seams only for collision runs. An
            # RGB run must never silently substitute a collision observation.
            if effective_profile != COLLISION_PROFILE:
                raise
            env = env_factory(stack_size=stack_size, step_frames=step_frames)
        if replay_class is None:
            replay = ReplayBuffer(
                capacity=selected_capacity,
                seed=seed,
                num_actions=num_actions,
                observation_shape=observation_shape_value,
                **({"store_discounts": True} if n_step > 1 else {}),
            )
        else:
            replay = replay_class(
                capacity=selected_capacity,
                seed=seed,
                num_actions=num_actions,
                stack_size=stack_size,
                **(
                    {"observation_profile": effective_profile}
                    if effective_profile == GRAY_PROFILE
                    else {}
                ),
            )

        def _network_factory(actions: int) -> torch.nn.Module:
            try:
                return AtariCnnQNetwork(
                    actions,
                    dueling=dueling_effective,
                    input_channels=observation_shape_value[0],
                    input_size=model_input_size,
                )
            except TypeError:
                # Old model factories remain valid for legacy collision runs;
                # RGB must fail rather than construct an 84x84 network.
                if effective_profile != COLLISION_PROFILE:
                    raise
                return AtariCnnQNetwork(
                    actions,
                    dueling=dueling_effective,
                    input_channels=observation_shape_value[0],
                )

        agent = DoubleDQNAgent(
            num_actions=num_actions,
            gamma=float(model_config.get("gamma", 0.99)),
            learning_rate=lr_effective,
            device=chosen_device,
            seed=seed,
            observation_shape=observation_shape_value,
            network_factory=_network_factory,
        )
        if resume_from is not None:
            _load_checkpoint(
                agent,
                Path(resume_from),
                observation_profile=effective_profile,
                stack_size=stack_size,
            )
        start_time = time.perf_counter()
        observation, reset_info = env.reset(seed=game_seed(0))
        palette_reset = getattr(replay, "reset_palette_ids", None)
        palette_frame = reset_info.get("native_palette_indices")
        if callable(palette_reset) and palette_frame is not None:
            palette_reset(palette_frame)
        accumulator = None
        if n_step > 1:
            from .n_step import NStepAccumulator

            accumulator = NStepAccumulator(n_step, gamma=agent.gamma)
    except Exception as error:
        writer.update_status(
            "failed",
            gate="fail",
            step=0,
            episode=0,
            message=f"{type(error).__name__}: {error}",
        )
        with contextlib.suppress(Exception):
            env.close()
        raise
    episode = 0
    episode_reward = 0.0
    episode_shaped = 0.0
    episode_frames = 0
    episode_deaths = 0
    episode_patterns = 0
    episode_spawns = 0
    episode_kills = 0
    total_deaths = 0
    total_patterns = 0
    total_spawns = 0
    total_kills = 0
    last_shattered = 0
    last_score = 0.0
    last_episode_reward = 0.0
    last_episode_frames = 0
    last_loss: float | None = None
    optimizer_step_start = agent.optimizer_steps
    last_update_step = optimizer_step_start
    last_metric_reward = 0.0
    last_metric_shaped = 0.0
    last_metric_frames = 0
    last_metric_deaths = 0
    last_metric_patterns = 0
    last_metric_spawns = 0
    last_metric_kills = 0
    best_score = 0.0
    # Phase 1/3 diagnostics: lightweight, bounded, JSON-safe.
    action_counts = [0] * max(1, num_actions)
    recent_rewards: deque[float] = deque(maxlen=512)
    last_td_mean = 0.0
    last_td_std = 0.0
    last_target_std = 0.0
    last_q_std = 0.0
    last_q_mean = 0.0
    last_target_mean = 0.0
    last_pre_clip_grad_norm = 0.0
    last_dead_units = 0.0
    last_diagnostic_optimizer_step: int | None = None
    target_sync_count_start = (
        _checkpoint_counter(resume_payload, "target_sync_count")
        if resume_payload is not None
        else 0
    )
    target_sync_count = target_sync_count_start
    completed_episodes_since_log: list[dict[str, object]] = []
    episode_return_state = EpisodeReturnState(episode=1)
    epsilon = epsilon_at(
        global_step_start,
        start=float(exploration_config.get("epsilon_start", 1.0)),
        final=float(exploration_config.get("epsilon_final", 0.1)),
        decay_steps=decay_effective,
    )
    step = 0
    paused_reported = False
    stopped = False

    def build_environment(settings: Mapping[str, Any]) -> CNNImageDDQNEnv:
        kwargs = {
            "stack_size": stack_size,
            "step_frames": int(settings["native_step_frames"]),
            "difficulty": int(settings["difficulty"]),
            "patterns": bool(settings["patterns"]),
            "powerups": bool(settings["powerups"]),
            "observation_profile": effective_profile,
        }
        try:
            return env_factory(**kwargs)
        except TypeError:
            if effective_profile != COLLISION_PROFILE:
                raise
            return env_factory(
                stack_size=stack_size,
                step_frames=int(settings["native_step_frames"]),
            )

    def service_commands() -> None:
        nonlocal env, observation, selected_game, episode_reward, episode_frames
        nonlocal last_episode_reward, last_episode_frames
        nonlocal episode_deaths, episode_patterns, episode_spawns
        nonlocal episode_kills, episode_shaped, last_shattered, last_score
        nonlocal episode_return_state
        if control is None:
            return
        for command in control.drain_commands():
            if command.kind in {"save", "config-save"}:
                if command.path is None:
                    control.emit(
                        TrainingEvent(command.kind, False, "Save path is missing")
                    )
                    continue
                try:
                    checkpoint_path, relative = _checkpoint_path(writer, command.path)
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        _checkpoint_payload(
                            agent,
                            step=resumed_step + step,
                            seed=seed,
                            observation_profile=effective_profile,
                            stack_size=stack_size,
                            run_id=run_id,
                            n_step=n_step,
                            reward_profile=reward_profile,
                            target_sync_count=target_sync_count,
                            source=source_provenance,
                        ),
                        checkpoint_path,
                    )
                    if relative is not None:
                        writer.record_checkpoint(relative)
                    control.emit(
                        TrainingEvent(
                            command.kind,
                            True,
                            f"Saved {checkpoint_path.name}",
                            checkpoint_path,
                            agent.optimizer_steps,
                        )
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as error:
                    control.emit(
                        TrainingEvent(command.kind, False, str(error), command.path)
                    )
                continue

            if command.kind == "load":
                if command.path is None:
                    control.emit(TrainingEvent("load", False, "Model path is missing"))
                    continue
                try:
                    payload = _load_checkpoint(
                        agent,
                        command.path,
                        observation_profile=effective_profile,
                        stack_size=stack_size,
                        expected_n_step=n_step,
                    )
                    control.emit(
                        TrainingEvent(
                            "load",
                            True,
                            f"Loaded {command.path.name} at step "
                            f"{int(payload.get('step', resumed_step + step)):,}",
                            command.path,
                            agent.optimizer_steps,
                        )
                    )
                except (
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    KeyError,
                ) as error:
                    control.emit(TrainingEvent("load", False, str(error), command.path))
                continue

            if command.kind == "config":
                try:
                    selected_game = _native_game_config(
                        command.config, default_step_frames=step_frames
                    )
                    env.close()
                    env = build_environment(selected_game)
                    observation, reset_info = env.reset(seed=game_seed(episode))
                    if replay_class is not None:
                        palette_reset = getattr(replay, "reset_palette_ids", None)
                        palette_frame = reset_info.get("native_palette_indices")
                        if callable(palette_reset) and palette_frame is not None:
                            palette_reset(palette_frame)
                        else:
                            replay.reset(observation)
                    episode_reward = 0.0
                    episode_frames = 0
                    episode_deaths = 0
                    episode_patterns = 0
                    episode_spawns = 0
                    episode_kills = 0
                    episode_shaped = 0.0
                    last_shattered = 0
                    last_score = 0.0
                    episode_return_state = EpisodeReturnState(episode=episode + 1)
                    last_episode_reward = 0.0
                    last_episode_frames = 0
                    control.emit(
                        TrainingEvent(
                            "config",
                            True,
                            "Game configuration applied; native settings "
                            "start on the next episode",
                        )
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as error:
                    control.emit(TrainingEvent("config", False, str(error)))

    def service_pause() -> None:
        nonlocal paused_reported
        if control is None:
            return
        while control.paused and not control.stopped:
            if not paused_reported:
                writer.update_status(
                    "paused",
                    step=step,
                    episode=episode,
                    message="paused by dashboard",
                )
                paused_reported = True
            service_commands()
            control.wait()
        if paused_reported and not control.stopped:
            writer.update_status(
                "running",
                step=step,
                episode=episode,
                message="training resumed",
            )
            paused_reported = False

    try:
        for step in range(1, steps + 1):
            service_pause()
            service_commands()
            if control is not None and control.stopped:
                stopped = True
                break
            global_step = resumed_step + step
            epsilon = epsilon_at(
                global_step,
                start=float(exploration_config.get("epsilon_start", 1.0)),
                final=float(exploration_config.get("epsilon_final", 0.1)),
                decay_steps=decay_effective,
            )
            action = int(agent.select_action(observation, epsilon, rng=rng))
            next_observation, reward, terminated, truncated, info = env.step(action)
            current_game_seed = game_seed(episode)
            seed_steps[current_game_seed] = seed_steps.get(current_game_seed, 0) + 1
            done = bool(terminated or truncated)
            if 0 <= action < len(action_counts):
                action_counts[action] += 1
            recent_rewards.append(float(reward))
            frames_now = int(info["native_frames_advanced"])
            flags_now = int(info.get("native_event_flags", 0) or 0)
            shattered_tmp = int(info.get("native_shattered", 0) or 0)
            kills_tmp = max(0, shattered_tmp - last_shattered)
            score_tmp = float(info.get("native_score", 0.0) or 0.0)
            score_delta_tmp = max(0.0, score_tmp - last_score)
            agent_delta_tmp = max(
                0.0, score_delta_tmp - UNCONTROLLED_SCORE_PER_ENEMY * kills_tmp
            )
            if reward_profile != "survival-v1":
                components = weighted_components(
                    info.get("native_reward_terms"), reward_profile
                )
                if components[0] != float(reward):
                    raise ValueError(
                        "native reward components disagree with survival reward"
                    )
                component_totals += components
                train_reward = float(components.sum())
            elif shaping_enabled:
                train_reward = (
                    shaping_weights["survival_per_frame"] * frames_now
                    + shaping_weights["score_weight"] * agent_delta_tmp
                    + shaping_weights["uncontrolled_score_weight"]
                    * UNCONTROLLED_SCORE_PER_ENEMY
                    * kills_tmp
                    - shaping_weights["death_penalty"] * (1.0 if terminated else 0.0)
                )
            else:
                train_reward = float(reward)
                component_totals[0] += float(reward)
            if accumulator is None:
                palette_add = getattr(replay, "add_palette_ids", None)
                palette_frame = info.get("native_palette_indices")
                if callable(palette_add) and palette_frame is not None:
                    palette_add(palette_frame, action, train_reward, done)
                else:
                    replay.add(
                        _observation_to_uint8(observation, observation_shape_value),
                        action,
                        train_reward,
                        _observation_to_uint8(
                            next_observation, observation_shape_value
                        ),
                        done,
                    )
            else:
                stored_observation = _observation_to_uint8(
                    observation, observation_shape_value
                )
                stored_next_observation = _observation_to_uint8(
                    next_observation, observation_shape_value
                )
                transitions = accumulator.add(
                    stored_observation,
                    action,
                    train_reward,
                    stored_next_observation,
                    bool(terminated),
                    bool(truncated),
                )
                if step == steps:
                    transitions.extend(accumulator.flush())
                for transition in transitions:
                    replay.add(
                        transition.observation,
                        transition.action,
                        transition.reward,
                        transition.next_observation,
                        transition.done,
                        discount=transition.bootstrap_discount,
                    )
            observation = next_observation
            episode_reward += float(reward)
            episode_shaped += float(train_reward)
            episode_frames += frames_now
            episode_return_state, completed_episode = record_episode_step(
                episode_return_state,
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                survival_frames=frames_now,
            )
            flags = int(info.get("native_event_flags", 0) or 0)
            if flags & EVENT_DEATH:
                episode_deaths += 1
                total_deaths += 1
            if flags & EVENT_PATTERN:
                episode_patterns += 1
                total_patterns += 1
            if flags & EVENT_SPAWN:
                episode_spawns += 1
                total_spawns += 1
            last_shattered = shattered_tmp
            last_score = score_tmp
            episode_kills += kills_tmp
            total_kills += kills_tmp

            if (
                len(replay) >= batch_size
                and (
                    len(replay) >= warmup_steps if n_step == 1 else step >= warmup_steps
                )
                and step % update_every == 0
            ):
                diagnostics_requested = _diagnostics_requested(
                    step=step,
                    update_every=update_every,
                    log_interval=log_interval,
                    steps=steps,
                    optimizer_step=agent.optimizer_steps,
                    optimizer_step_start=optimizer_step_start,
                )
                batch = (
                    replay.sample_packed(batch_size)
                    if effective_profile != COLLISION_PROFILE
                    else replay.sample(batch_size)
                )
                update = agent.update(batch, diagnostics=diagnostics_requested)
                last_update_step = update.optimizer_step
                if update.diagnostics_sampled:
                    last_diagnostic_optimizer_step = update.optimizer_step
                    last_loss = update.loss
                    last_td_mean = float(update.td_error_mean)
                    last_td_std = float(update.td_error_std)
                    last_target_std = float(update.target_std)
                    last_q_std = float(update.q_std)
                    last_q_mean = float(update.mean_q)
                    last_target_mean = float(update.mean_target)
                    last_pre_clip_grad_norm = float(update.pre_clip_grad_norm)
                if update.optimizer_step % target_sync_interval == 0:
                    agent.sync_target()
                    target_sync_count += 1

            finished_reward: float | None = None
            finished_frames: int | None = None
            finished_deaths: int | None = None
            finished_patterns: int | None = None
            finished_spawns: int | None = None
            finished_kills: int | None = None
            finished_shaped: float | None = None
            if done:
                finished_reward = episode_reward
                finished_frames = episode_frames
                finished_deaths = episode_deaths
                finished_patterns = episode_patterns
                finished_spawns = episode_spawns
                finished_kills = episode_kills
                finished_shaped = episode_shaped
                episode += 1
                if completed_episode is None:
                    raise RuntimeError("completed episode record is missing")
                completed_episodes_since_log.append(
                    {
                        **completed_episode,
                        "game_seed": current_game_seed,
                        "global_step": global_step,
                        "shaped_reward": float(finished_shaped),
                        "deaths": int(finished_deaths),
                        "patterns_survived": int(finished_patterns),
                        "enemies_spawned": int(finished_spawns),
                        "enemies_killed": int(finished_kills),
                    }
                )
                observation, reset_info = env.reset(
                    seed=game_seed(episode),
                )
                palette_reset = getattr(replay, "reset_palette_ids", None)
                palette_frame = reset_info.get("native_palette_indices")
                if callable(palette_reset) and palette_frame is not None:
                    palette_reset(palette_frame)
                episode_reward = 0.0
                episode_shaped = 0.0
                episode_frames = 0
                episode_deaths = 0
                episode_patterns = 0
                episode_spawns = 0
                episode_kills = 0
                last_shattered = 0
                last_score = 0.0

            elapsed = max(time.perf_counter() - start_time, 1e-9)
            if step == 1 or step % log_interval == 0 or step == steps:
                last_metric_reward = (
                    finished_reward if finished_reward is not None else episode_reward
                )
                last_metric_frames = (
                    finished_frames if finished_frames is not None else episode_frames
                )
                last_metric_deaths = (
                    finished_deaths if finished_deaths is not None else episode_deaths
                )
                last_metric_patterns = (
                    finished_patterns
                    if finished_patterns is not None
                    else episode_patterns
                )
                last_metric_spawns = (
                    finished_spawns if finished_spawns is not None else episode_spawns
                )
                last_metric_kills = (
                    finished_kills if finished_kills is not None else episode_kills
                )
                last_metric_shaped = (
                    finished_shaped if finished_shaped is not None else episode_shaped
                )
                best_score = max(best_score, last_metric_reward, 0.0)
                mix = reward_mix_stats(list(recent_rewards))
                balance = action_balance_ratio(action_counts)
                least_used = min(
                    range(len(action_counts)),
                    key=lambda i: action_counts[i],
                )
                with contextlib.suppress(RuntimeError, ValueError):
                    last_dead_units = float(agent.dead_unit_fraction(observation))
                metrics: dict[str, object] = {
                    "step": step,
                    "global_environment_step": global_step,
                    "segment_step": step,
                    "training_seeds_visited": len(seed_steps),
                    "training_seed_count": training_seed_count,
                    "episode": episode,
                    "reward": last_metric_reward,
                    "shaped_reward": last_metric_shaped,
                    "shaping_enabled": shaping_enabled,
                    "native_reward_profile": reward_profile,
                    "reward_component_totals": component_totals.tolist(),
                    "survival_frames": last_metric_frames,
                    "episode_length": last_metric_frames / 60.0,
                    "enemies_killed": last_metric_kills,
                    "enemies_destroyed": last_metric_kills,
                    "total_kills": total_kills,
                    "deaths": last_metric_deaths,
                    "total_deaths": total_deaths,
                    "patterns_survived": last_metric_patterns,
                    "total_patterns": total_patterns,
                    "enemies_spawned": last_metric_spawns,
                    "total_spawns": total_spawns,
                    "policy_entropy": _epsilon_entropy(epsilon, num_actions),
                    "best_score": best_score,
                    "epsilon": epsilon,
                    "epsilon_decay_effective": decay_effective,
                    "epsilon_decay_configured": decay_configured,
                    "throughput": step / elapsed,
                    "replay_size": len(replay),
                    "optimizer_step": last_update_step,
                    "global_optimizer_step": last_update_step,
                    "segment_optimizer_step": (last_update_step - optimizer_step_start),
                    "reward_zero_share": mix["zero_share"],
                    "reward_mean": mix["mean"],
                    "reward_mean_nonzero": mix["mean_nonzero"],
                    "reward_std": mix["std"],
                    "action_balance": balance,
                    "least_used_action": least_used,
                    "action_counts": list(action_counts),
                    "dead_units": last_dead_units,
                    "target_sync_count": target_sync_count,
                    "segment_target_sync_count": (
                        target_sync_count - target_sync_count_start
                    ),
                    "reward_scope": (
                        "completed-episode"
                        if finished_reward is not None
                        else "partial-episode"
                    ),
                    "completed_episodes": list(completed_episodes_since_log),
                }
                metrics["diagnostic_sample_cadence"] = DIAGNOSTIC_SAMPLE_CADENCE
                metrics["diagnostic_sampled"] = (
                    last_diagnostic_optimizer_step is not None
                )
                if last_diagnostic_optimizer_step is not None:
                    metrics.update(
                        {
                            "diagnostic_optimizer_step": (
                                last_diagnostic_optimizer_step
                            ),
                            "loss": last_loss,
                            "td_error_mean": last_td_mean,
                            "td_error_std": last_td_std,
                            "target_std": last_target_std,
                            "q_std": last_q_std,
                            "q_mean": last_q_mean,
                            "target_mean": last_target_mean,
                            "q_target_mean_delta": last_q_mean - last_target_mean,
                            "pre_clip_grad_norm": last_pre_clip_grad_norm,
                            "gradient_clipped": (
                                last_pre_clip_grad_norm > MAX_GRAD_NORM
                            ),
                        }
                    )
                writer.append_metrics(metrics)
                completed_episodes_since_log.clear()
                writer.update_status(
                    "running",
                    step=step,
                    episode=episode,
                    current_metrics=metrics,
                )

            if step in snapshot_steps and step != steps:
                relative = f"checkpoints/step-{global_step}.pt"
                snapshot_path = writer.paths.relative(relative)
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    _checkpoint_payload(
                        agent,
                        step=global_step,
                        seed=seed,
                        observation_profile=effective_profile,
                        stack_size=stack_size,
                        run_id=run_id,
                        n_step=n_step,
                        reward_profile=reward_profile,
                        target_sync_count=target_sync_count,
                        source=source_provenance,
                    ),
                    snapshot_path,
                )
                writer.record_checkpoint(relative)

        if stopped:
            stopped_metrics = {
                "step": step,
                "global_environment_step": resumed_step + step,
                "segment_step": step,
                "episode": episode,
                "reward": last_metric_reward,
                "survival_frames": last_metric_frames,
                "loss": last_loss,
                "epsilon": epsilon,
                "throughput": step / max(time.perf_counter() - start_time, 1e-9),
                "replay_size": len(replay),
                "optimizer_step": last_update_step,
                "global_optimizer_step": last_update_step,
                "segment_optimizer_step": last_update_step - optimizer_step_start,
            }
            report = {
                "result": "stopped_by_dashboard",
                "quality_gate": "warn",
                "final_metrics": stopped_metrics,
            }
            writer.finalize(gate="warn", report=report, state="stopped")
            return writer.paths.root

        checkpoint_relative = f"checkpoints/step-{global_step_end}.pt"
        checkpoint_path = writer.paths.relative(checkpoint_relative)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            _checkpoint_payload(
                agent,
                step=global_step_end,
                seed=seed,
                observation_profile=effective_profile,
                stack_size=stack_size,
                run_id=run_id,
                n_step=n_step,
                reward_profile=reward_profile,
                target_sync_count=target_sync_count,
                source=source_provenance,
            ),
            checkpoint_path,
        )
        writer.record_checkpoint(checkpoint_relative)

        evaluation_inner = _evaluate(
            agent,
            env,
            seed=eval_base_seed,
            episodes=eval_episodes,
            max_steps=eval_steps,
            seed_offset=10_000,
        )
        evaluation_holdout = _evaluate(
            agent,
            env,
            seed=eval_base_seed,
            episodes=eval_episodes,
            max_steps=eval_steps,
            seed_offset=20_000,
        )
        greedy_action_counts = [
            int(inner) + int(holdout)
            for inner, holdout in zip(
                evaluation_inner.get("action_counts", []),
                evaluation_holdout.get("action_counts", []),
                strict=True,
            )
        ]
        greedy_action_balance = action_balance_ratio(greedy_action_counts)
        evaluation_censored_share = max(
            float(evaluation_inner.get("censored_share", 0.0)),
            float(evaluation_holdout.get("censored_share", 0.0)),
        )
        least_used_final = min(
            range(len(action_counts)), key=lambda i: action_counts[i]
        )
        try:
            counterfactual = _counterfactual_eval(
                agent,
                env,
                seed=eval_base_seed,
                episodes=min(eval_episodes, 4),
                max_steps=eval_steps,
                forced_action=int(least_used_final),
            )
        except (RuntimeError, ValueError):
            counterfactual = {
                "episodes": 0,
                "forced_action": int(least_used_final),
                "mean_reward": 0.0,
                "rewards": [],
                "baseline_rewards": [],
                "reward_deltas": [],
                "mean_reward_delta": 0.0,
                "paired": paired_reward_deltas([], []),
            }
        train_mean = float(evaluation_inner.get("mean_reward", 0.0))
        holdout_mean = float(evaluation_holdout.get("mean_reward", 0.0))
        gap_denominator = abs(train_mean) + 1.0
        train_holdout_gap = abs(train_mean - holdout_mean) / gap_denominator
        final_mix = reward_mix_stats(list(recent_rewards))
        final_balance = action_balance_ratio(action_counts)
        gate, gate_reasons = decide_gate(
            steps=steps,
            warmup_steps=warmup_steps,
            target_sync_interval=target_sync_interval,
            decay_configured=decay_configured,
            decay_effective=decay_effective,
            balance_ratio=final_balance,
            zero_share=float(final_mix["zero_share"]),
            train_mean=train_mean,
            holdout_mean=holdout_mean,
            updates=int(last_update_step - optimizer_step_start),
            target_sync_count=target_sync_count - target_sync_count_start,
            evaluation_censored_share=evaluation_censored_share,
        )
        evaluation = {
            "protocol": "frozen-train-holdout-v1",
            "observation_profile": effective_profile,
            "observation_shape": list(observation_shape_value),
            "stack_size": stack_size,
            "inner": evaluation_inner,
            "holdout": evaluation_holdout,
            "train_holdout_gap": train_holdout_gap,
            "evaluation_censored_share": evaluation_censored_share,
            "counterfactual": counterfactual,
            # The reported best is one greedy-policy episode from this
            # frozen final evaluation — the training-curve maximum was
            # collected under exploration by an older, unsaved policy.
            "best_eval_episode": best_greedy_episode(
                {"inner": evaluation_inner, "holdout": evaluation_holdout}
            ),
            # Back-compat: top-level fields mirror the inner eval so old
            # readers keep working without knowing about holdout split.
            "episodes": evaluation_inner.get("episodes"),
            "mean_reward": evaluation_inner.get("mean_reward"),
            "mean_survival_frames": evaluation_inner.get("mean_survival_frames"),
            "rewards": evaluation_inner.get("rewards"),
            "seeds": evaluation_inner.get("seeds"),
        }
        best_eval = evaluation["best_eval_episode"]
        final_metrics = {
            "step": steps,
            "global_environment_step": global_step_end,
            "segment_step": steps,
            "episode": episode,
            "reward": last_metric_reward,
            "survival_frames": last_metric_frames,
            "episode_length": last_metric_frames / 60.0,
            "enemies_killed": 0.0,
            "policy_entropy": _epsilon_entropy(epsilon, num_actions),
            "best_score": best_score,
            "best_eval_reward": float(best_eval["eval_reward"])
            if isinstance(best_eval, dict)
            else 0.0,
            "best_eval_episode": best_eval,
            "epsilon": epsilon,
            "epsilon_decay_effective": decay_effective,
            "throughput": steps / max(time.perf_counter() - start_time, 1e-9),
            "replay_size": len(replay),
            "optimizer_step": last_update_step,
            "global_optimizer_step": last_update_step,
            "segment_optimizer_step": last_update_step - optimizer_step_start,
            "reward_zero_share": float(final_mix["zero_share"]),
            "action_balance": final_balance,
            "least_used_action": int(least_used_final),
            "action_counts": list(action_counts),
            "greedy_eval_action_counts": greedy_action_counts,
            "greedy_eval_action_balance": greedy_action_balance,
            "dead_units": last_dead_units,
            "train_holdout_gap": train_holdout_gap,
            "target_sync_count": target_sync_count,
            "segment_target_sync_count": (target_sync_count - target_sync_count_start),
            "diagnostic_sample_cadence": DIAGNOSTIC_SAMPLE_CADENCE,
            "diagnostic_sampled": last_diagnostic_optimizer_step is not None,
        }
        if last_diagnostic_optimizer_step is not None:
            final_metrics.update(
                {
                    "diagnostic_optimizer_step": last_diagnostic_optimizer_step,
                    "loss": last_loss,
                    "td_error_mean": last_td_mean,
                    "td_error_std": last_td_std,
                    "target_std": last_target_std,
                    "q_std": last_q_std,
                    "q_mean": last_q_mean,
                    "target_mean": last_target_mean,
                    "q_target_mean_delta": last_q_mean - last_target_mean,
                    "pre_clip_grad_norm": last_pre_clip_grad_norm,
                    "gradient_clipped": last_pre_clip_grad_norm > MAX_GRAD_NORM,
                }
            )
        report = {
            "result": "bounded_baseline_complete",
            "native_reward_contract": native_reward_contract,
            "reward_component_totals": component_totals.tolist(),
            "quality_gate": gate,
            "gate_reasons": gate_reasons,
            "reason": "; ".join(gate_reasons) if gate_reasons else "steady",
            "final_metrics": final_metrics,
            "evaluation": evaluation,
            "checkpoint": checkpoint_relative,
            "diagnostics": {
                "training_seeds_visited": len(seed_steps),
                "training_seed_steps": seed_steps,
                "training_seed_pool_coverage": (
                    len(seed_steps) / len(pool) if pool else None
                ),
                "reward_mix": final_mix,
                "action_balance": final_balance,
                "action_counts": list(action_counts),
                "greedy_eval_action_counts": greedy_action_counts,
                "greedy_eval_action_balance": greedy_action_balance,
                "decay_configured": decay_configured,
                "decay_effective": decay_effective,
                "warmup_steps": warmup_steps,
                "target_sync_interval": target_sync_interval,
                "target_sync_count": target_sync_count,
                "optimizer_updates": int(last_update_step - optimizer_step_start),
                "segment_optimizer_updates": int(
                    last_update_step - optimizer_step_start
                ),
                "global_optimizer_updates": int(last_update_step),
                "diagnostic_sample_cadence": DIAGNOSTIC_SAMPLE_CADENCE,
                "diagnostic_optimizer_step": last_diagnostic_optimizer_step,
                "segment_target_sync_count": (
                    target_sync_count - target_sync_count_start
                ),
            },
        }
        writer.finalize(gate=gate, evaluation=evaluation, report=report)
    except Exception as error:
        writer.update_status(
            "failed",
            gate="fail",
            step=int(locals().get("step", 0)),
            episode=episode,
            message=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        env.close()

    return writer.paths.root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=_positive_int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=_nonnegative_int, default=DEFAULT_SEED)
    parser.add_argument("--training-seed-count", type=_positive_int, default=None)
    parser.add_argument("--evaluation-seed", type=_nonnegative_int, default=None)
    parser.add_argument(
        "--stack-size", type=_positive_int, choices=(1, 2, 4, 8), default=4
    )
    parser.add_argument(
        "--observation-profile",
        choices=OBSERVATION_PROFILES,
        default=None,
        help="observation profile; omitted uses the variant config (legacy collision)",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--replay-capacity", type=_positive_int, default=None)
    parser.add_argument("--n-step", type=int, choices=(1, 3), default=1)
    parser.add_argument(
        "--warmup-steps", type=_positive_int, default=DEFAULT_WARMUP_STEPS
    )
    parser.add_argument(
        "--update-every", type=_positive_int, default=DEFAULT_UPDATE_EVERY
    )
    parser.add_argument(
        "--target-sync-interval",
        type=_positive_int,
        default=DEFAULT_TARGET_SYNC_INTERVAL,
    )
    parser.add_argument(
        "--log-interval", type=_positive_int, default=DEFAULT_LOG_INTERVAL
    )
    parser.add_argument(
        "--eval-episodes", type=_positive_int, default=DEFAULT_EVAL_EPISODES
    )
    parser.add_argument("--eval-steps", type=_positive_int, default=DEFAULT_EVAL_STEPS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--dueling", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epsilon-decay-steps", type=_positive_int, default=None)
    parser.add_argument(
        "--shaping", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--w-survival", type=float, default=None)
    parser.add_argument("--w-death", type=float, default=None)
    parser.add_argument(
        "--reward-profile", choices=tuple(PROFILES), default="survival-v1"
    )
    parser.add_argument("--w-score", type=float, default=None)
    parser.add_argument("--w-uncontrolled", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = train_run(
        history_root=args.history_root,
        run_id=args.run_id,
        steps=args.steps,
        seed=args.seed,
        training_seed_count=args.training_seed_count,
        evaluation_seed=args.evaluation_seed,
        stack_size=args.stack_size,
        observation_profile=args.observation_profile,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        n_step=args.n_step,
        warmup_steps=args.warmup_steps,
        update_every=args.update_every,
        target_sync_interval=args.target_sync_interval,
        log_interval=args.log_interval,
        eval_episodes=args.eval_episodes,
        eval_steps=args.eval_steps,
        device=args.device,
        resume_from=args.resume_from,
        dueling=args.dueling,
        learning_rate=args.learning_rate,
        epsilon_decay_steps=args.epsilon_decay_steps,
        shaping=args.shaping,
        reward_profile=args.reward_profile,
        w_survival=args.w_survival,
        w_death=args.w_death,
        w_score=args.w_score,
        w_uncontrolled=args.w_uncontrolled,
    )
    print(json.dumps({"run_root": str(path), "dashboard_variant": VARIANT_ID}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
