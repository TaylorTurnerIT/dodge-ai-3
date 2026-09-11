"""Compact uint8 replay storage for image-only Double-DQN transitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import IMAGE_SHAPE


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    """A validated, owned minibatch sampled from replay storage."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray

    def __post_init__(self) -> None:
        observations = np.asarray(self.observations)
        next_observations = np.asarray(self.next_observations)
        actions = np.asarray(self.actions)
        rewards = np.asarray(self.rewards)
        dones = np.asarray(self.dones)
        if observations.ndim != 4:
            raise ValueError("observations must have shape (N, C, 84, 84)")
        if next_observations.shape != observations.shape:
            raise ValueError("next_observations must match observations shape")
        batch_size = observations.shape[0]
        if actions.shape != (batch_size,):
            raise ValueError("actions must have shape (N,)")
        if rewards.shape != (batch_size,):
            raise ValueError("rewards must have shape (N,)")
        if dones.shape != (batch_size,):
            raise ValueError("dones must have shape (N,)")
        if observations.shape[1] < 1 or observations.shape[2:] != (84, 84):
            raise ValueError("observations must have shape (N, C, 84, 84)")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "next_observations", next_observations)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "dones", dones)

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.observations.shape[1:])


def _frame_to_uint8(frame: object, expected_shape: tuple[int, int, int]) -> np.ndarray:
    value = np.asarray(frame)
    if value.shape != expected_shape:
        raise ValueError(f"frame must have shape {expected_shape}, got {value.shape}")
    if not np.issubdtype(value.dtype, np.number):
        raise ValueError("frame must be numeric")
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 255.0):
        raise ValueError("frame values must be finite and in [0, 255]")
    if float(value.max(initial=0.0)) <= 1.0:
        value = value * np.float32(255.0)
    return np.rint(value).astype(np.uint8, copy=True)


class ReplayBuffer:
    """Fixed-capacity ring buffer that keeps image frames as uint8."""

    def __init__(
        self,
        capacity: int,
        *,
        seed: int | None = None,
        num_actions: int | None = None,
        observation_shape: tuple[int, int, int] = IMAGE_SHAPE,
    ) -> None:
        if isinstance(capacity, bool) or capacity < 1:
            raise ValueError("capacity must be positive")
        shape = tuple(int(value) for value in observation_shape)
        if len(shape) != 3 or shape[0] < 1 or shape[1:] != (84, 84):
            raise ValueError("observation_shape must be (channels, 84, 84)")
        if num_actions is not None and (
            isinstance(num_actions, bool) or num_actions < 1
        ):
            raise ValueError("num_actions must be positive when provided")
        self.capacity = int(capacity)
        self.observation_shape = shape
        self.num_actions = None if num_actions is None else int(num_actions)
        self._rng = np.random.default_rng(seed)
        self._observations = np.empty(
            (self.capacity, *self.observation_shape), dtype=np.uint8
        )
        self._next_observations = np.empty_like(self._observations)
        self._actions = np.empty(self.capacity, dtype=np.int64)
        self._rewards = np.empty(self.capacity, dtype=np.float32)
        self._dones = np.empty(self.capacity, dtype=np.bool_)
        self._size = 0
        self._next_index = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        observation: object,
        action: int,
        reward: float,
        next_observation: object,
        done: bool,
    ) -> None:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer")
        action_value = int(action)
        if self.num_actions is not None and not 0 <= action_value < self.num_actions:
            raise ValueError(f"action must be between 0 and {self.num_actions - 1}")
        reward_value = float(reward)
        if not np.isfinite(reward_value):
            raise ValueError("reward must be finite")
        index = self._next_index
        self._observations[index] = _frame_to_uint8(
            observation, self.observation_shape
        )
        self._next_observations[index] = _frame_to_uint8(
            next_observation, self.observation_shape
        )
        self._actions[index] = action_value
        self._rewards[index] = reward_value
        self._dones[index] = bool(done)
        self._next_index = (index + 1) % self.capacity
        self._size = min(self.capacity, self._size + 1)

    def sample(self, batch_size: int) -> ReplayBatch:
        if isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be positive")
        if batch_size > self._size:
            raise ValueError("cannot sample more transitions than are stored")
        indices = self._rng.choice(self._size, size=int(batch_size), replace=False)
        return ReplayBatch(
            observations=self._observations[indices].copy(),
            actions=self._actions[indices].copy(),
            rewards=self._rewards[indices].copy(),
            next_observations=self._next_observations[indices].copy(),
            dones=self._dones[indices].copy(),
        )


__all__ = ["ReplayBatch", "ReplayBuffer"]
