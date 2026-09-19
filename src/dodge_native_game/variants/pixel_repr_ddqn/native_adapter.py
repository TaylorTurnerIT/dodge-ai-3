"""Pixel-only access to the native Dodge RGB environment.

The existing CNN image environment exposes useful native diagnostics in its
``info`` mapping.  Those diagnostics are intentionally not part of this
variant's input boundary.  :class:`PixelNativeAdapter` is the small audited
surface used by the collector: callers can reset a lane and receive pixels,
or apply an action and receive pixels plus ordinary Gym termination values.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..cnn_image_ddqn.env import CNNImageDDQNEnv
from ..cnn_image_ddqn.pixels import RGB_PROFILE
from .scenario import ScenarioConfig, load_scenario

ACTION_COUNT = 9
NATIVE_SEED_MAX = 32_767
RGB_SHAPE = (3, 128, 128)


class _GymPixelEnvironment(Protocol):
    """The deliberately tiny Gym surface needed by the adapter."""

    def reset(self, *, seed: int) -> tuple[object, object]: ...

    def step(self, action: int) -> tuple[object, float, bool, bool, object]: ...

    def close(self) -> None: ...


def _native_terms(info: object) -> np.ndarray | None:
    """Retain only the native 6-term reward vector from ``info``."""

    if not isinstance(info, dict):
        return None
    terms = info.get("native_reward_terms")
    if terms is None:
        return None
    values = np.asarray(terms, dtype=np.float32).reshape(-1)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        return None
    return np.ascontiguousarray(values)


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


def _default_environment(
    scenario: ScenarioConfig | None = None,
) -> CNNImageDDQNEnv:
    """Construct the native RGB environment at stack size one.

    The ordinary path remains the existing CNN environment constructor.  A
    configured scenario owns a separately constructed native batch object,
    which is injected through the audited legacy Gym wrapper so this variant
    does not alter the legacy environment's defaults.
    """

    if scenario is None:
        return CNNImageDDQNEnv(stack_size=1, observation_profile=RGB_PROFILE)

    from ...batch import NativeBatchEnvironment

    native = NativeBatchEnvironment(
        step_frames=4,
        full_state=False,
        pixels=True,
        board=False,
        collision_image=False,
        **scenario.native_kwargs(),
    )
    return CNNImageDDQNEnv(
        stack_size=1,
        step_frames=4,
        difficulty=scenario.difficulty,
        patterns=scenario.patterns_enabled,
        powerups=scenario.powerups_enabled,
        observation_profile=RGB_PROFILE,
        native_environment=native,
    )


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
        scenario: ScenarioConfig | Path | str | None = None,
    ) -> None:
        if environment is not None and environment_factory is not None:
            raise ValueError("provide environment or environment_factory, not both")
        if isinstance(scenario, ScenarioConfig):
            scenario_config = scenario
        elif scenario is None:
            scenario_config = None
        else:
            scenario_config = load_scenario(scenario)
        if scenario_config is not None and (
            environment is not None or environment_factory is not None
        ):
            raise ValueError(
                "scenario cannot be combined with an injected environment"
            )
        if environment is not None:
            self._environment = environment
        elif environment_factory is not None:
            self._environment = environment_factory()
        else:
            self._environment = _default_environment(scenario_config)
        self._scenario = scenario_config
        self._closed = False
        self._last_terms: np.ndarray | None = None

    @property
    def scenario(self) -> ScenarioConfig | None:
        """Resolved scenario used to construct the native environment, if any."""

        return self._scenario

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
        observation, reward, terminated, truncated, info = result
        terminated_value = bool(terminated)
        truncated_value = bool(truncated)
        if terminated_value and truncated_value:
            raise ValueError(
                "native environment cannot be both terminated and truncated"
            )
        self._last_terms = _native_terms(info)
        return (
            _owned_rgb(observation),
            float(reward),
            terminated_value,
            truncated_value,
        )

    @property
    def last_reward_terms(self) -> np.ndarray:
        """Native 6-term reward vector of the latest decision (P7 §2).

        Order: survival, death, pickups, enemy_deaths, edge, corner.
        Only this field is retained from the native info mapping; every
        other diagnostic stays discarded.  Raises when the wrapped
        environment does not supply terms (fixture injections).
        """

        if self._last_terms is None:
            raise ValueError("native reward terms unavailable")
        return self._last_terms.copy()

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
