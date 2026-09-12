"""Bounded n-step transition accumulation for collision DDQN.

The environment emits one-step transitions, while a learner may train on a
bounded return.  :class:`NStepAccumulator` keeps only the current episode's
pending transitions and emits owned transitions as soon as their return is
complete.  Episode boundaries are explicit: a terminal transition masks the
bootstrap term, whereas a truncation ends the accumulation window but keeps
the transition non-terminal so its value can bootstrap.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


def _validate_n_step(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("n_step must be a positive integer")
    value = int(value)
    if value < 1:
        raise ValueError("n_step must be a positive integer")
    return value


def _validate_gamma(value: object) -> float:
    try:
        gamma = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("gamma must be finite and between 0 and 1") from error
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and between 0 and 1")
    return gamma


def _validate_bool(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _owned_observation(value: object, name: str) -> np.ndarray:
    try:
        observation = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an array-like observation") from error
    if observation.dtype == np.dtype(object):
        raise ValueError(f"{name} must contain numeric or structured data")
    return np.array(observation, copy=True)


@dataclass(frozen=True, slots=True)
class NStepTransition:
    """An owned transition with a bounded return.

    ``reward`` is the discounted return over ``n_steps`` environment steps.
    ``bootstrap_discount`` is ``gamma ** n_steps`` even for terminal
    transitions; the learner applies ``done`` as the bootstrap mask.  Keeping
    the discount separate makes terminal suffixes and truncated episodes
    unambiguous in replay.
    """

    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool
    bootstrap_discount: float
    n_steps: int = 1

    def __post_init__(self) -> None:
        observation = _owned_observation(self.observation, "observation")
        next_observation = _owned_observation(
            self.next_observation, "next_observation"
        )
        if observation.shape != next_observation.shape:
            raise ValueError("next_observation must match observation shape")
        if isinstance(self.action, bool) or not isinstance(
            self.action, (int, np.integer)
        ):
            raise TypeError("action must be an integer")
        try:
            reward = float(self.reward)
            bootstrap_discount = float(self.bootstrap_discount)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("reward and bootstrap_discount must be numeric") from error
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        if not np.isfinite(bootstrap_discount) or not 0.0 <= bootstrap_discount <= 1.0:
            raise ValueError("bootstrap_discount must be finite and between 0 and 1")
        n_steps = _validate_n_step(self.n_steps)
        done = _validate_bool(self.done, "done")
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "next_observation", next_observation)
        object.__setattr__(self, "action", int(self.action))
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "done", done)
        object.__setattr__(self, "bootstrap_discount", bootstrap_discount)
        object.__setattr__(self, "n_steps", n_steps)

@dataclass(slots=True)
class _PendingTransition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminated: bool
    boundary: bool


class NStepAccumulator:
    """Accumulate one episode into bounded discounted transitions.

    ``add`` returns every transition that became ready after the supplied
    one-step transition.  A regular stream emits one item once ``n_step``
    transitions are available.  A terminal or truncated step flushes all
    remaining suffixes immediately.  Calling ``flush`` at a segment boundary
    emits pending partial windows without marking them terminal.

    The accumulator is deliberately independent of replay storage.  A caller
    can insert each returned :class:`NStepTransition` with its
    ``bootstrap_discount`` as replay metadata, or inspect/transform the owned
    transitions first.
    """

    def __init__(self, n_step: int = 1, *, gamma: float = 0.99) -> None:
        self.n_step = _validate_n_step(n_step)
        self.gamma = _validate_gamma(gamma)
        self._pending: deque[_PendingTransition] = deque()
        self._observation_shape: tuple[int, ...] | None = None

    def __len__(self) -> int:
        """Return the number of one-step transitions waiting for a return."""

        return len(self._pending)

    @property
    def pending(self) -> int:
        """Number of transitions currently buffered but not yet emitted."""

        return len(self._pending)

    def add(
        self,
        observation: object,
        action: int,
        reward: float,
        next_observation: object,
        terminated: bool = False,
        truncated: bool = False,
        *,
        done: bool | None = None,
    ) -> list[NStepTransition]:
        """Add one environment step and return newly ready transitions.

        ``done`` is accepted as a compatibility alias for a terminal step.
        New callers should pass ``terminated`` and ``truncated`` separately;
        truncation closes this episode but its emitted transitions have
        ``done=False`` and therefore still bootstrap.
        """

        if done is not None:
            if terminated or truncated:
                raise TypeError("done cannot be combined with terminated/truncated")
            terminated = done
        terminated_value = _validate_bool(terminated, "terminated")
        truncated_value = _validate_bool(truncated, "truncated")
        observation_value = _owned_observation(observation, "observation")
        next_observation_value = _owned_observation(
            next_observation, "next_observation"
        )
        if observation_value.shape != next_observation_value.shape:
            raise ValueError("next_observation must match observation shape")
        if self._observation_shape is None:
            self._observation_shape = tuple(int(v) for v in observation_value.shape)
        elif observation_value.shape != self._observation_shape:
            raise ValueError(
                "observation shape changed within an n-step accumulation stream"
            )
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer")
        try:
            reward_value = float(reward)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("reward must be numeric") from error
        if not np.isfinite(reward_value):
            raise ValueError("reward must be finite")

        self._pending.append(
            _PendingTransition(
                observation=observation_value,
                action=int(action),
                reward=reward_value,
                next_observation=next_observation_value,
                terminated=terminated_value,
                boundary=terminated_value or truncated_value,
            )
        )

        if terminated_value or truncated_value:
            return self.flush()
        if len(self._pending) >= self.n_step:
            return [self._emit_oldest()]
        return []

    def flush(self) -> list[NStepTransition]:
        """Emit all pending suffixes and clear the current episode window."""

        emitted: list[NStepTransition] = []
        while self._pending:
            emitted.append(self._emit_oldest())
        return emitted

    def _emit_oldest(self) -> NStepTransition:
        if not self._pending:
            raise RuntimeError("cannot emit from an empty n-step accumulator")
        window = list(self._pending)[: self.n_step]
        reward_value = 0.0
        gamma_power = 1.0
        horizon = 0
        done = False
        last = window[0]
        for pending in window:
            reward_value += gamma_power * pending.reward
            horizon += 1
            last = pending
            done = done or pending.terminated
            if pending.boundary:
                break
            gamma_power *= self.gamma

        self._pending.popleft()
        # The power is computed from the realized horizon rather than from
        # queue length, so terminal/truncated suffixes receive gamma**k too.
        bootstrap_discount = self.gamma**horizon
        return NStepTransition(
            observation=window[0].observation,
            action=window[0].action,
            reward=reward_value,
            next_observation=last.next_observation,
            done=done,
            bootstrap_discount=bootstrap_discount,
            n_steps=horizon,
        )


__all__ = ["NStepAccumulator", "NStepTransition"]
