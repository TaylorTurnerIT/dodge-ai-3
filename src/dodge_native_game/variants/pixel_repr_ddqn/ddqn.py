"""Double DQN on frozen LeWM latents (P8, first P7 branch).

Policies read a 576-dim state (3 most recent projected 192-dim latents, the
same history the planner uses) through the shared live-input boundary and
learn from the pinned P7 reward (survival frames + terminal death, unit
weights).  The world model stays frozen: no encoder gradients, ever.  This
module implements the branch mechanics; training budgets and schedules live
in the frozen P9 protocol.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Callable
from typing import Any, Final

import numpy as np
import torch
from torch import nn

from .live_inputs import history_batch, model_palette

STATE_DIM: Final[int] = 576
LATENT_DIM: Final[int] = 192
HISTORY_FRAMES: Final[int] = 3
ACTION_COUNT: Final[int] = 9
P7_REWARD_WEIGHTS: Final[tuple[float, ...]] = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0)

__all__ = [
    "ACTION_COUNT",
    "HISTORY_FRAMES",
    "LATENT_DIM",
    "P7_REWARD_WEIGHTS",
    "STATE_DIM",
    "DDQNAgent",
    "DuelingMLP",
    "ReplayBuffer",
    "collect_episode",
    "encode_state",
    "p7_reward",
]


def p7_reward(terms: Any) -> float:
    """Unshaped P7 reward: survival frames + terminal death (unit weights)."""

    values = np.asarray(terms, dtype=np.float32).reshape(-1)
    if values.shape != (6,):
        raise ValueError(f"P7 reward needs 6 native terms, got {values.shape}")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("P7 reward terms must be finite")
    weights = np.asarray(P7_REWARD_WEIGHTS, dtype=np.float32)
    return float(values @ weights)


class DuelingMLP(nn.Module):
    """576 -> 256 -> 256 dueling head with 9 action values."""

    def __init__(
        self, state_dim: int = STATE_DIM, hidden: int = 256
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, ACTION_COUNT)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        features = self.trunk(states)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class ReplayBuffer:
    """Fixed-capacity ring buffer of flat latent transitions."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = capacity
        self.states: deque[np.ndarray] = deque(maxlen=capacity)
        self.actions: deque[int] = deque(maxlen=capacity)
        self.rewards: deque[float] = deque(maxlen=capacity)
        self.next_states: deque[np.ndarray | None] = deque(maxlen=capacity)
        self.done: deque[bool] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.states)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray | None,
        done: bool,
    ) -> None:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"state must be ({STATE_DIM},), got {state.shape}")
        if next_state is not None:
            next_state = np.asarray(next_state, dtype=np.float32).reshape(-1)
            if next_state.shape != (STATE_DIM,):
                raise ValueError("next_state must match state shape")
        if done and next_state is not None:
            raise ValueError("terminal transitions carry no next state")
        if not done and next_state is None:
            raise ValueError("non-terminal transitions need a next state")
        if isinstance(action, bool) or not isinstance(action, int):
            raise TypeError("action must be an integer")
        if not 0 <= action < ACTION_COUNT:
            raise ValueError(f"action must be in [0, {ACTION_COUNT - 1}]")
        self.states.append(np.ascontiguousarray(state))
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.next_states.append(
            None if next_state is None else np.ascontiguousarray(next_state)
        )
        self.done.append(bool(done))

    def sample(
        self, batch_size: int, rng: random.Random
    ) -> dict[str, torch.Tensor]:
        if batch_size < 1 or len(self) < batch_size:
            raise ValueError("not enough transitions to sample")
        indices = rng.sample(range(len(self)), batch_size)
        states = np.stack([self.states[i] for i in indices])
        next_states = np.stack(
            [
                self.next_states[i]
                if self.next_states[i] is not None
                else np.zeros(STATE_DIM, dtype=np.float32)
                for i in indices
            ]
        )
        return {
            "states": torch.from_numpy(states),
            "actions": torch.tensor(
                [self.actions[i] for i in indices], dtype=torch.int64
            ),
            "rewards": torch.tensor(
                [self.rewards[i] for i in indices], dtype=torch.float32
            ),
            "next_states": torch.from_numpy(next_states),
            "done": torch.tensor(
                [self.done[i] for i in indices], dtype=torch.float32
            ),
        }


class DDQNAgent:
    """Double DQN with a separate target network (V10)."""

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        sync_interval: int = 100,
        device: torch.device | str = "cpu",
    ) -> None:
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        self.online = DuelingMLP()
        self.target = DuelingMLP()
        self.target.load_state_dict(self.online.state_dict())
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=learning_rate
        )
        self.gamma = gamma
        self.sync_interval = sync_interval
        self.device = torch.device(device)
        self.online.to(self.device)
        self.target.to(self.device)
        self.updates = 0

    def act(
        self, state: np.ndarray, epsilon: float, rng: random.Random
    ) -> int:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        flat = np.asarray(state, dtype=np.float32).reshape(-1)
        if flat.shape != (STATE_DIM,):
            raise ValueError(f"state must be ({STATE_DIM},), got {flat.shape}")
        if rng.random() < epsilon:
            return rng.randrange(ACTION_COUNT)
        self.online.eval()
        with torch.no_grad():
            values = self.online(
                torch.from_numpy(flat.reshape(1, -1)).to(self.device)
            )
        return int(torch.argmax(values, dim=-1).item())

    def update(self, batch: dict[str, torch.Tensor]) -> float:
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        done = batch["done"].to(self.device)
        self.online.train()
        current = (
            self.online(states)
            .gather(1, actions.unsqueeze(1))
            .squeeze(1)
        )
        with torch.no_grad():
            next_online = self.online(next_states)
            best = torch.argmax(next_online, dim=-1, keepdim=True)
            next_value = (
                self.target(next_states).gather(1, best).squeeze(1)
            )
            target = rewards + self.gamma * (1.0 - done) * next_value
        loss = nn.functional.mse_loss(current, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.sync_interval == 0:
            self.sync()
        return float(loss.item())

    def sync(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def state_dict(self) -> dict[str, Any]:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
            "gamma": self.gamma,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.online.load_state_dict(payload["online"])
        self.target.load_state_dict(payload["target"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.updates = int(payload["updates"])
        self.gamma = float(payload["gamma"])


@torch.no_grad()
def encode_state(
    model: torch.nn.Module,
    frames: deque[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    """Encode trailing history frames into one flat 576-dim state."""

    palette = model_palette(model)
    history, _, _ = history_batch(frames, device, palette)
    latents = model.encode(history)
    state = latents[0, -HISTORY_FRAMES:, :].reshape(-1)
    return state.detach().to("cpu").to(torch.float32).numpy().copy()


def collect_episode(
    make_adapter: Callable[[], Any],
    seed: int,
    agent: DDQNAgent,
    model: torch.nn.Module,
    rng: random.Random,
    *,
    epsilon: float,
    device: torch.device,
    history_size: int = HISTORY_FRAMES,
    max_decisions: int = 128,
) -> dict[str, Any]:
    """Roll one epsilon-greedy episode; record P7 transitions.

    Terminal deaths end bootstrapping (``done=True``, no next state);
    truncations keep the next state (bootstrap).  Returns transitions plus
    the familiar survival/outcome/count envelope.
    """

    adapter = make_adapter()
    try:
        frame = adapter.reset(seed=seed)
        frames: deque[np.ndarray] = deque(
            [np.asarray(frame, dtype=np.uint8)] * (history_size + 1),
            maxlen=history_size + 1,
        )
        transitions: list[
            tuple[np.ndarray, int, float, np.ndarray | None, bool]
        ] = []
        survived = 0
        outcome = "truncated"
        masked_pixels = 0
        projected_pixels = 0
        palette = model_palette(model)
        for _step in range(max_decisions):
            _, newly_masked, newly_projected = history_batch(
                frames, device, palette
            )
            masked_pixels += newly_masked
            projected_pixels += newly_projected
            state = encode_state(model, frames, device)
            action = agent.act(state, epsilon, rng)
            frame, _reward, terminated, truncated = adapter.step(action)
            reward = p7_reward(adapter.last_reward_terms)
            frames.append(np.asarray(frame, dtype=np.uint8))
            if terminated:
                outcome = "terminated"
                transitions.append((state, action, reward, None, True))
                break
            next_state = encode_state(model, frames, device)
            transitions.append((state, action, reward, next_state, False))
            survived += 1
            if truncated:
                outcome = "truncated"
                break
        else:
            survived = max_decisions
        return {
            "seed": seed,
            "survived": survived,
            "outcome": outcome,
            "masked_pixels": masked_pixels,
            "projected_pixels": projected_pixels,
            "transitions": transitions,
        }
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()
