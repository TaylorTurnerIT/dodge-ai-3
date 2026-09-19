from __future__ import annotations

import random
from collections import deque

import numpy as np
import pytest
import torch

from dodge_native_game.variants.pixel_repr_ddqn.ddqn import (
    ACTION_COUNT,
    STATE_DIM,
    DDQNAgent,
    DuelingMLP,
    ReplayBuffer,
    collect_episode,
    encode_state,
    p7_reward,
)


def test_p7_reward_uses_survival_plus_death_only() -> None:
    assert p7_reward([4.0, 0.0, 0.0, 0.0, -0.0, -0.0]) == 4.0
    assert p7_reward([2.0, -1.0, 5.0, 3.0, -1.0, -1.0]) == 1.0
    with pytest.raises(ValueError, match="6 native terms"):
        p7_reward([1.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        p7_reward([1.0, 0.0, 0.0, 0.0, 0.0, float("nan")])


def test_dueling_head_shapes_and_gradients() -> None:
    head = DuelingMLP()
    values = head(torch.zeros(5, STATE_DIM))
    assert values.shape == (5, ACTION_COUNT)
    values.sum().backward()
    assert all(
        p.grad is not None for p in head.parameters() if p.requires_grad
    )


def _state(fill: float) -> np.ndarray:
    return np.full(STATE_DIM, fill, dtype=np.float32)


def test_replay_buffer_validates_and_samples_mixed_batches() -> None:
    buffer = ReplayBuffer(capacity=4)
    buffer.push(_state(1.0), 3, 4.0, _state(2.0), False)
    buffer.push(_state(2.0), 0, -1.0, None, True)
    assert len(buffer) == 2
    batch = buffer.sample(2, random.Random(0))
    assert batch["states"].shape == (2, STATE_DIM)
    assert batch["actions"].tolist() in ([3, 0], [0, 3])
    assert batch["rewards"].shape == (2,)
    assert batch["next_states"].shape == (2, STATE_DIM)
    assert batch["done"].tolist() in ([0.0, 1.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="terminal transitions"):
        buffer.push(_state(0.0), 0, 0.0, _state(0.0), True)
    with pytest.raises(ValueError, match="next state"):
        buffer.push(_state(0.0), 0, 0.0, None, False)
    with pytest.raises(ValueError, match="action must be"):
        buffer.push(_state(0.0), 9, 0.0, _state(0.0), False)
    for index in range(4):
        buffer.push(_state(float(index)), 0, 0.0, _state(0.0), False)
    assert len(buffer) == 4


def test_agent_act_is_greedy_or_random() -> None:
    agent = DDQNAgent(device="cpu")
    state = _state(0.0)
    greedy = agent.act(state, 0.0, random.Random(1))
    assert greedy == agent.act(state, 0.0, random.Random(2))
    assert 0 <= greedy < ACTION_COUNT
    seen = {
        agent.act(state, 1.0, random.Random(seed)) for seed in range(50)
    }
    assert len(seen) > 1
    with pytest.raises(ValueError, match="epsilon"):
        agent.act(state, 2.0, random.Random(0))
    with pytest.raises(ValueError, match="STATE_DIM|576"):
        agent.act(np.zeros(4, dtype=np.float32), 0.0, random.Random(0))


def _zero(agent: DDQNAgent) -> None:
    for net in (agent.online, agent.target):
        for parameter in net.parameters():
            parameter.detach().zero_()


def test_update_uses_double_dqn_target() -> None:
    agent = DDQNAgent(gamma=0.5, learning_rate=0.0, device="cpu")
    _zero(agent)
    with torch.no_grad():
        agent.online.advantage.bias[2] = 10.0
        agent.target.value.bias[0] = 1.0
        agent.target.advantage.bias[2] = 4.0
        agent.target.advantage.bias[5] = 100.0
    batch = {
        "states": torch.zeros(1, STATE_DIM),
        "actions": torch.tensor([2]),
        "rewards": torch.tensor([3.0]),
        "next_states": torch.zeros(1, STATE_DIM),
        "done": torch.tensor([0.0]),
    }
    loss = agent.update(batch)
    # Online argmax is action 2; target evaluates action 2:
    # Q(s,2) = 10 - 10/9; V' = 1 + 4 - 104/9; target = 3 + 0.5 * V'.
    current = 10.0 - 10.0 / 9.0
    next_value = 1.0 + 4.0 - 104.0 / 9.0
    assert loss == pytest.approx((current - (3.0 + 0.5 * next_value)) ** 2)
    assert agent.updates == 1


def test_target_syncs_on_schedule_and_checkpoint_roundtrips() -> None:
    agent = DDQNAgent(sync_interval=2, device="cpu")
    batch = {
        "states": torch.zeros(2, STATE_DIM),
        "actions": torch.tensor([1, 4]),
        "rewards": torch.tensor([1.0, 2.0]),
        "next_states": torch.zeros(2, STATE_DIM),
        "done": torch.tensor([0.0, 1.0]),
    }
    before = [
        p.detach().clone() for p in agent.target.parameters()
    ]
    agent.update(batch)
    assert all(
        torch.equal(a, b)
        for a, b in zip(agent.target.parameters(), before, strict=True)
    )
    agent.update(batch)
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            agent.target.parameters(), agent.online.parameters(), strict=True
        )
    )
    payload = agent.state_dict()
    clone = DDQNAgent(device="cpu")
    clone.load_state_dict(payload)
    assert clone.updates == 2
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            clone.online.parameters(), agent.online.parameters(), strict=True
        )
    )


class _FlatModel(torch.nn.Module):
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        batch, time = pixels.shape[:2]
        return torch.ones(batch, time, 192)


def test_encode_state_flattens_trailing_latents() -> None:
    frames = deque(
        [np.zeros((3, 128, 128), dtype=np.uint8)] * 4, maxlen=4
    )
    state = encode_state(_FlatModel(), frames, torch.device("cpu"))
    assert state.shape == (STATE_DIM,)
    assert (state == 1.0).all()


class _TermsAdapter:
    """Scripted adapter serving native reward terms like the real env."""

    def __init__(self, die_at: int | None = None) -> None:
        self._die_at = die_at
        self._step = 0
        frame = np.zeros((3, 128, 128), dtype=np.uint8)
        frame[:] = np.asarray((41, 173, 255), dtype=np.uint8).reshape(3, 1, 1)
        self._frame = frame
        self._terms = np.zeros(6, dtype=np.float32)

    def reset(self, seed: int):
        del seed
        self._step = 0
        return self._frame

    def step(self, action: int):
        del action
        done = self._die_at is not None and self._step == self._die_at
        self._terms = np.asarray(
            [4.0, -1.0 if done else 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        self._step += 1
        return self._frame, 0.0, done, False

    @property
    def last_reward_terms(self) -> np.ndarray:
        return self._terms.copy()

    def close(self) -> None:
        pass


def test_collect_episode_records_p7_transitions() -> None:
    agent = DDQNAgent(device="cpu")
    report = collect_episode(
        lambda: _TermsAdapter(die_at=2), 7, agent, _FlatModel(),
        random.Random(0), epsilon=1.0, device=torch.device("cpu"),
        max_decisions=8,
    )
    assert report["outcome"] == "terminated"
    assert report["survived"] == 2
    assert len(report["transitions"]) == 3
    *_, (state, action, reward, next_state, done) = report["transitions"]
    assert (reward, next_state, done) == (3.0, None, True)
    assert state.shape == (STATE_DIM,) and 0 <= action < ACTION_COUNT
    first = report["transitions"][0]
    assert first[2] == 4.0 and first[3] is not None and first[4] is False


def test_collect_episode_bootstraps_on_truncation() -> None:
    agent = DDQNAgent(device="cpu")
    report = collect_episode(
        _TermsAdapter, 7, agent, _FlatModel(), random.Random(0),
        epsilon=1.0, device=torch.device("cpu"), max_decisions=4,
    )
    assert report["outcome"] == "truncated"
    assert len(report["transitions"]) == 4
    assert all(done is False for *_, done in report["transitions"])
    assert all(t[3] is not None for t in report["transitions"])
