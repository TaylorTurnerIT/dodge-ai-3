"""Pixel-only access to the native Dodge RGB environment.

The existing CNN image environment exposes useful native diagnostics in its
``info`` mapping.  Those diagnostics are intentionally not part of this
variant's input boundary.  :class:`PixelNativeAdapter` is the small audited
surface used by the collector: callers can reset a lane and receive pixels,
or apply an action and receive pixels plus ordinary Gym termination values.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from ..cnn_image_ddqn.env import CNNImageDDQNEnv
from ..cnn_image_ddqn.pixels import RGB_PROFILE

ACTION_COUNT = 9
NATIVE_SEED_MAX = 32_767
RGB_SHAPE = (3, 128, 128)


class _GymPixelEnvironment(Protocol):
    """The deliberately tiny Gym surface needed by the adapter."""

    def reset(self, *, seed: int) -> tuple[object, object]: ...

    def step(self, action: int) -> tuple[object, float, bool, bool, object]: ...

    def close(self) -> None: ...


def _validate_seed(seed: int | np.integer) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    value = int(seed)
    if not 0 <= value <= NATIVE_SEED_MAX:
        raise ValueError(f"seed must be between 0 and {NATIVE_SEED_MAX}")
    return value


def _validate_action(action: int | np.integer) -> int:
    if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
        raise TypeError("action must be an integer")
    value = int(action)
    if not 0 <= value < ACTION_COUNT:
        raise ValueError(f"action must be between 0 and {ACTION_COUNT - 1}")
    return value


def _owned_rgb(value: object) -> np.ndarray:
    pixels = np.asarray(value)
    if pixels.shape != RGB_SHAPE or pixels.dtype != np.uint8:
        raise ValueError(
            "native RGB observation must have shape (3, 128, 128) and dtype uint8"
        )
    return np.array(pixels, dtype=np.uint8, copy=True, order="C")


def _default_environment() -> CNNImageDDQNEnv:
    """Construct the existing native RGB environment at stack size one."""

    return CNNImageDDQNEnv(stack_size=1, observation_profile=RGB_PROFILE)


class PixelNativeAdapter:
    """Expose only native RGB pixels, actions, reward, and episode boundaries.

    ``environment`` and ``environment_factory`` are optional injection points
    for bounded fixture tests.  Production collection uses the existing
    :class:`CNNImageDDQNEnv` with ``native-rgb-v1`` and ``stack_size=1``.
    The fifth Gym return value is deliberately discarded in both ``reset``
    and ``step``; callers cannot accidentally train on native diagnostics.
    """

    def __init__(
        self,
        environment: _GymPixelEnvironment | None = None,
        *,
        environment_factory: Callable[[], _GymPixelEnvironment] | None = None,
    ) -> None:
        if environment is not None and environment_factory is not None:
            raise ValueError("provide environment or environment_factory, not both")
        if environment is not None:
            self._environment = environment
        elif environment_factory is not None:
            self._environment = environment_factory()
        else:
            self._environment = _default_environment()
        self._closed = False

    @property
    def action_count(self) -> int:
        return ACTION_COUNT

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return RGB_SHAPE

    def reset(self, *, seed: int | np.integer) -> np.ndarray:
        """Reset one native episode and return its owned RGB framebuffer."""

        self._ensure_open()
        native_seed = _validate_seed(seed)
        result = self._environment.reset(seed=native_seed)
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("native environment reset must return (observation, info)")
        observation, _info = result
        return _owned_rgb(observation)

    def step(self, action: int | np.integer) -> tuple[np.ndarray, float, bool, bool]:
        """Apply one action and discard the native info mapping."""

        self._ensure_open()
        native_action = _validate_action(action)
        result = self._environment.step(native_action)
        if not isinstance(result, tuple) or len(result) != 5:
            raise ValueError(
                "native environment step must return "
                "(observation, reward, terminated, truncated, info)"
            )
        observation, reward, terminated, truncated, _info = result
        terminated_value = bool(terminated)
        truncated_value = bool(truncated)
        if terminated_value and truncated_value:
            raise ValueError(
                "native environment cannot be both terminated and truncated"
            )
        return (
            _owned_rgb(observation),
            float(reward),
            terminated_value,
            truncated_value,
        )

    def close(self) -> None:
        if not self._closed:
            self._environment.close()
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("pixel native adapter is closed")

    def __enter__(self) -> PixelNativeAdapter:
        self._ensure_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


NativePixelAdapter = PixelNativeAdapter

__all__ = [
    "ACTION_COUNT",
    "NATIVE_SEED_MAX",
    "NativePixelAdapter",
    "PixelNativeAdapter",
    "RGB_SHAPE",
]
