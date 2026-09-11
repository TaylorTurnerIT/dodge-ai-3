"""Gymnasium environment for the native collision-image DDQN variant."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .image import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    NativeBatchResultError,
    NativeCollisionImageUnavailable,
    collision_image_from_result,
)
from .temporal import TemporalFrameStack

ACTION_COUNT = 9
DEFAULT_STEP_FRAMES = 4
DEFAULT_STACK_SIZE = 4
NATIVE_SEED_MAX = 32_767


class NativeBatchBoundary(Protocol):
    """Minimal public surface required from the shared native Python wrapper."""

    def reset_batch(self, seeds: object, *, startup: bool = False) -> object: ...

    def step_batch(self, actions: object) -> object: ...

    def close(self) -> None: ...


def _result_field(result: object, name: str) -> object:
    if isinstance(result, Mapping):
        try:
            return result[name]
        except KeyError as error:
            raise NativeBatchResultError(
                f"native batch result did not expose {name!r}"
            ) from error
    try:
        return getattr(result, name)
    except AttributeError as error:
        raise NativeBatchResultError(
            f"native batch result did not expose {name!r}"
        ) from error


def _lane_scalar(result: object, name: str) -> Any:
    value = np.asarray(_result_field(result, name))
    if value.size != 1:
        raise NativeBatchResultError(
            f"native batch result field {name!r} must contain one lane; "
            f"got shape {value.shape}"
        )
    return value.reshape(-1)[0].item()


def _optional_lane_int(result: object, name: str, default: int) -> int:
    """Read one optional integer lane; test doubles may not provide it."""

    try:
        return int(_lane_scalar(result, name))
    except (NativeBatchResultError, ValueError, TypeError):
        return default


def _optional_lane_float(result: object, name: str, default: float) -> float:
    """Read one optional float lane; test doubles may not provide it."""

    try:
        return float(_lane_scalar(result, name))
    except (NativeBatchResultError, ValueError, TypeError):
        return default


def _native_seed(seed: int | np.integer | None, rng: np.random.Generator) -> int:
    if seed is None:
        return int(rng.integers(0, NATIVE_SEED_MAX + 1))
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer or None")
    value = int(seed)
    if value < 0 or value > NATIVE_SEED_MAX:
        raise ValueError(f"seed must be between 0 and {NATIVE_SEED_MAX}")
    return value


class CNNImageDDQNEnv(gym.Env[np.ndarray, int]):
    """One native Dodge lane with a grayscale temporal CNN observation.

    Rust remains responsible for game state, collision semantics, reward,
    termination, and the collision-image contents. Python only validates the
    native image, maintains the requested frame history, and translates the
    batch result to Gymnasium's API.
    """

    metadata: dict[str, list[str]] = {"render_modes": []}

    def __init__(
        self,
        *,
        stack_size: int = DEFAULT_STACK_SIZE,
        step_frames: int = DEFAULT_STEP_FRAMES,
        difficulty: int = 2,
        patterns: bool = True,
        powerups: bool = True,
        native_environment: NativeBatchBoundary | None = None,
    ) -> None:
        if isinstance(step_frames, bool) or not isinstance(step_frames, int):
            raise TypeError("step_frames must be an integer")
        if step_frames < 1:
            raise ValueError("step_frames must be positive")
        if isinstance(difficulty, bool) or not isinstance(difficulty, int):
            raise TypeError("difficulty must be an integer")
        if difficulty not in (1, 2, 3):
            raise ValueError("difficulty must be 1, 2, or 3")
        if not isinstance(patterns, bool) or not isinstance(powerups, bool):
            raise TypeError("patterns and powerups must be booleans")

        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = spaces.Box(
            low=np.float32(0.0),
            high=np.float32(1.0),
            shape=(stack_size, FRAME_HEIGHT, FRAME_WIDTH),
            dtype=np.float32,
        )
        self._frames = TemporalFrameStack(stack_size)
        self._step_frames = step_frames
        self._difficulty = difficulty
        self._patterns = patterns
        self._powerups = powerups
        self._native = native_environment or self._make_native(
            step_frames,
            difficulty=difficulty,
            patterns=patterns,
            powerups=powerups,
        )
        self._episode_ended = False
        self._closed = False

    @staticmethod
    def _make_native(
        step_frames: int,
        *,
        difficulty: int = 2,
        patterns: bool = True,
        powerups: bool = True,
    ) -> NativeBatchBoundary:
        # Keep the rendered pixel and board payloads off. The variant consumes
        # only the native collision_image field when the boundary provides it.
        from ...batch import NativeBatchEnvironment

        return NativeBatchEnvironment(
            step_frames=step_frames,
            full_state=False,
            pixels=False,
            board=False,
            difficulty=difficulty,
            patterns_enabled=patterns,
            powerups_enabled=powerups,
            collision_image=True,
        )

    @property
    def stack_size(self) -> int:
        """Configured number of temporal frames."""

        return self._frames.stack_size

    @property
    def step_frames(self) -> int:
        """Native decision interval used by this environment."""

        return self._step_frames

    @property
    def frame_count(self) -> int:
        """Current number of frames held by the temporal deque."""

        return self._frames.frame_count

    def reset(
        self,
        *,
        seed: int | np.integer | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the native lane and fill every stack slot with its first image."""

        self._ensure_open()
        super().reset(seed=seed)
        native_seed = _native_seed(seed, self.np_random)
        startup = False
        if options:
            unknown = set(options) - {"startup"}
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(f"unsupported reset options: {names}")
            startup = bool(options.get("startup", False))
        result = self._native.reset_batch(
            np.asarray([native_seed], dtype=np.uint32), startup=startup
        )
        observation = self._frames.reset(collision_image_from_result(result))
        self._episode_ended = False
        return observation, self._info(result, native_seed=native_seed)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the native lane once and append exactly one image frame."""

        self._ensure_open()
        if self._episode_ended:
            raise RuntimeError("step() called after the native episode terminated")
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, {ACTION_COUNT - 1}]")

        result = self._native.step_batch(np.asarray([int(action)], dtype=np.uint8))
        observation = self._frames.append(collision_image_from_result(result))
        terminated = bool(_lane_scalar(result, "done"))
        self._episode_ended = terminated
        reward = float(_lane_scalar(result, "rewards"))
        return (
            observation,
            reward,
            terminated,
            False,
            self._info(result),
        )

    def render(self) -> None:
        """This image variant has no Python-side renderer."""

        return None

    def close(self) -> None:
        if not self._closed:
            self._native.close()
            self._closed = True

    def _info(
        self, result: object, *, native_seed: int | None = None
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "native_frame": int(_lane_scalar(result, "frames")),
            "native_frames_advanced": int(_lane_scalar(result, "frames_advanced")),
            "native_done": bool(_lane_scalar(result, "done")),
            "native_event_flags": _optional_lane_int(result, "event_flags", 0),
            "native_mode": _optional_lane_int(result, "modes", -1),
            "native_shattered": _optional_lane_int(result, "shattered", 0),
            "native_score": _optional_lane_float(result, "score", 0.0),
        }
        if native_seed is not None:
            info["native_seed"] = native_seed
        return info

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("environment is closed")

    def __enter__(self) -> CNNImageDDQNEnv:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "ACTION_COUNT",
    "CNNImageDDQNEnv",
    "DEFAULT_STACK_SIZE",
    "DEFAULT_STEP_FRAMES",
    "NativeBatchBoundary",
    "NativeBatchResultError",
    "NativeCollisionImageUnavailable",
]
