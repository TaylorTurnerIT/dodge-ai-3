"""Compact uint8 replay storage for image-only Double-DQN transitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import IMAGE_SHAPE, validate_observation_shape


def estimate_replay_storage_bytes(
    capacity: int,
    observation_shape: tuple[int, int, int] = IMAGE_SHAPE,
) -> int:
    """Return the bytes allocated by a dense uint8 replay ring.

    The estimate includes both image rings and the fixed-width transition
    metadata arrays. It performs no allocation and does not reduce capacity.
    """

    if isinstance(capacity, bool) or capacity < 1:
        raise ValueError("capacity must be positive")
    shape = validate_observation_shape(observation_shape)
    frame_bytes = int(np.prod(shape, dtype=np.int64))
    return int(
        int(capacity)
        * (
            2 * frame_bytes * np.dtype(np.uint8).itemsize
            + np.dtype(np.int64).itemsize
            + np.dtype(np.float32).itemsize
            + np.dtype(np.bool_).itemsize
        )
    )


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
            raise ValueError("observations must have shape (N, C, H, W)")
        if next_observations.shape != observations.shape:
            raise ValueError("next_observations must match observations shape")
        if observations.dtype != np.uint8 or next_observations.dtype != np.uint8:
            raise ValueError("observations and next_observations must have dtype uint8")
        batch_size = observations.shape[0]
        if actions.shape != (batch_size,):
            raise ValueError("actions must have shape (N,)")
        if rewards.shape != (batch_size,):
            raise ValueError("rewards must have shape (N,)")
        if dones.shape != (batch_size,):
            raise ValueError("dones must have shape (N,)")
        validate_observation_shape(
            tuple(int(value) for value in observations.shape[1:])
        )
        object.__setattr__(
            self, "observations", np.array(observations, dtype=np.uint8, copy=True)
        )
        object.__setattr__(
            self,
            "next_observations",
            np.array(next_observations, dtype=np.uint8, copy=True),
        )
        object.__setattr__(self, "actions", np.array(actions, copy=True))
        object.__setattr__(self, "rewards", np.array(rewards, copy=True))
        object.__setattr__(self, "dones", np.array(dones, copy=True))

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
        if num_actions is not None and (
            isinstance(num_actions, bool) or num_actions < 1
        ):
            raise ValueError("num_actions must be positive when provided")
        self.capacity = int(capacity)
        self.observation_shape = validate_observation_shape(shape)
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

    @property
    def estimated_storage_bytes(self) -> int:
        """Return the dense ring's current allocation size in bytes."""

        return estimate_replay_storage_bytes(self.capacity, self.observation_shape)

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
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_observations=self._next_observations[indices],
            dones=self._dones[indices],
        )


__all__ = ["ReplayBatch", "ReplayBuffer", "estimate_replay_storage_bytes"]
