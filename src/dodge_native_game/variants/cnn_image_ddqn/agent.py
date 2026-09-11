"""Composable Double-DQN learner for the image-only Dodge variant."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import IMAGE_SHAPE, AtariCnnQNetwork, to_float_observations
from .replay import ReplayBatch

NetworkFactory = Callable[[int], nn.Module]
MAX_GRAD_NORM: Final = 10.0

__all__ = ["DDQNUpdate", "DoubleDQNAgent", "MAX_GRAD_NORM"]


@dataclass(frozen=True, slots=True)
class DDQNUpdate:
    """Scalar diagnostics from one optimizer update."""

    loss: float
    mean_q: float
    mean_target: float
    pre_clip_grad_norm: float
    batch_size: int
    optimizer_step: int
    td_error_mean: float = 0.0
    td_error_std: float = 0.0
    target_std: float = 0.0
    q_std: float = 0.0


class DoubleDQNAgent:
    """Online/target-network Double-DQN primitive.

    Environment workers can own collection and replay insertion separately;
    this class only selects actions, samples no data, and updates from an
    explicit :class:`ReplayBatch`.
    """

    def __init__(
        self,
        num_actions: int,
        *,
        gamma: float = 0.99,
        learning_rate: float = 1e-4,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        network_factory: NetworkFactory = AtariCnnQNetwork,
        online_network: nn.Module | None = None,
        target_network: nn.Module | None = None,
        observation_shape: tuple[int, int, int] = IMAGE_SHAPE,
    ) -> None:
        if isinstance(num_actions, bool) or num_actions < 1:
            raise ValueError("num_actions must be a positive integer")
        if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be finite and between 0 and 1")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if (
            not isinstance(observation_shape, tuple)
            or len(observation_shape) != 3
            or observation_shape[0] < 1
            or tuple(observation_shape[1:]) != (84, 84)
        ):
            raise ValueError("observation_shape must be (channels, 84, 84)")
        self.observation_shape = tuple(int(item) for item in observation_shape)
        if online_network is None:
            if network_factory is AtariCnnQNetwork:
                online_network = network_factory(
                    int(num_actions), input_channels=self.observation_shape[0]
                )
            else:
                online_network = network_factory(int(num_actions))
        if not isinstance(online_network, nn.Module):
            raise TypeError("network_factory must return torch.nn.Module")
        if target_network is None:
            target_network = deepcopy(online_network)
        if not isinstance(target_network, nn.Module):
            raise TypeError("target_network must be torch.nn.Module")
        if target_network is online_network:
            raise ValueError("online_network and target_network must be distinct")

        self.num_actions = int(num_actions)
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.device = torch.device(device)
        self.online_network = online_network.to(self.device)
        self.target_network = target_network.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=self.learning_rate
        )
        for parameter in self.target_network.parameters():
            parameter.requires_grad_(False)
        self.target_network.eval()
        self._rng = np.random.default_rng(seed)
        self.optimizer_steps = 0

    def select_action(
        self,
        observations: object,
        epsilon: float,
        *,
        rng: np.random.Generator | None = None,
    ) -> int | np.ndarray:
        """Select greedy or random actions for one or many image observations."""

        epsilon_value = self._validate_epsilon(epsilon)
        tensor, single = self._observation_tensor(observations)
        with torch.no_grad():
            q_values = self._checked_q_values(self.online_network, tensor)
        greedy_actions = q_values.argmax(dim=1).cpu().numpy().astype(np.int64)

        random_source = self._rng if rng is None else rng
        random_mask = random_source.random(greedy_actions.shape) < epsilon_value
        if np.any(random_mask):
            greedy_actions[random_mask] = random_source.integers(
                0, self.num_actions, size=int(np.count_nonzero(random_mask))
            )
        if single:
            return int(greedy_actions[0])
        return greedy_actions

    def update(self, batch: ReplayBatch) -> DDQNUpdate:
        """Run one Huber-loss Double-DQN update from an explicit batch."""

        if not isinstance(batch, ReplayBatch):
            raise TypeError("batch must be a ReplayBatch")
        if batch.observation_shape != self.observation_shape:
            raise ValueError(
                "batch observation shape does not match the agent: "
                f"{batch.observation_shape} != {self.observation_shape}"
            )
        observations, _ = self._observation_tensor(batch.observations)
        next_observations, _ = self._observation_tensor(batch.next_observations)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(
            batch.rewards, dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)
        if actions.ndim != 1 or actions.shape[0] != batch.size:
            raise ValueError("batch actions must have shape (N,)")
        if torch.any(actions < 0) or torch.any(actions >= self.num_actions):
            raise ValueError("batch actions contain an out-of-range action")

        self.online_network.train()
        q_values = self._checked_q_values(self.online_network, observations)
        chosen_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_online_q_values = self._checked_q_values(
                self.online_network, next_observations
            )
            next_actions = next_online_q_values.argmax(dim=1, keepdim=True)
            next_target_q_values = self._checked_q_values(
                self.target_network, next_observations
            )
            next_q_values = next_target_q_values.gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * (1.0 - dones) * next_q_values

        loss = F.smooth_l1_loss(chosen_q_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(), max_norm=MAX_GRAD_NORM
        )
        self.optimizer.step()
        self.optimizer_steps += 1

        with torch.no_grad():
            td_errors = targets.detach() - chosen_q_values.detach()
            td_mean = float(td_errors.mean().cpu().item()) if td_errors.numel() else 0.0
            td_std = (
                float(td_errors.std().cpu().item()) if td_errors.numel() > 1 else 0.0
            )
            target_std = (
                float(targets.detach().std().cpu().item())
                if targets.numel() > 1
                else 0.0
            )
            q_std = (
                float(chosen_q_values.detach().std().cpu().item())
                if chosen_q_values.numel() > 1
                else 0.0
            )

        return DDQNUpdate(
            loss=float(loss.detach().cpu().item()),
            mean_q=float(chosen_q_values.detach().mean().cpu().item()),
            mean_target=float(targets.detach().mean().cpu().item()),
            pre_clip_grad_norm=float(pre_clip_grad_norm.detach().cpu().item()),
            batch_size=batch.size,
            optimizer_step=self.optimizer_steps,
            td_error_mean=td_mean,
            td_error_std=td_std,
            target_std=target_std,
            q_std=q_std,
        )

    @torch.no_grad()
    def sync_target(self) -> None:
        """Hard-copy online parameters into the target network."""

        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

    @torch.no_grad()
    def q_values_for(self, observations: object) -> np.ndarray:
        """Return raw Q-values for logging without affecting training."""

        tensor, _ = self._observation_tensor(observations)
        q_values = self._checked_q_values(self.online_network, tensor)
        return q_values.detach().cpu().numpy().astype(np.float64, copy=True)

    @torch.no_grad()
    def dead_unit_fraction(self, observations: object) -> float:
        """Share of trunk activations that are exactly zero.

        Returns 0.0 when the online network does not expose the expected
        Atari trunk. Used as a stuck-network probe, not a training signal.
        """

        try:
            tensor, _ = self._observation_tensor(observations)
            features = self.online_network.features  # type: ignore[attr-defined]
            trunk = features(tensor)
        except (AttributeError, ValueError, RuntimeError):
            return 0.0
        values = trunk.detach().cpu().numpy()
        if values.size == 0:
            return 0.0
        return float(np.mean(values == 0.0))

    def _checked_q_values(
        self, network: nn.Module, observations: torch.Tensor
    ) -> torch.Tensor:
        q_values = network(observations)
        if q_values.ndim != 2 or q_values.shape != (
            observations.shape[0],
            self.num_actions,
        ):
            raise ValueError(
                "network must return shape "
                f"(N, {self.num_actions}), got {tuple(q_values.shape)}"
            )
        return q_values

    def _observation_tensor(self, observations: object) -> tuple[torch.Tensor, bool]:
        tensor = (
            observations
            if isinstance(observations, torch.Tensor)
            else torch.as_tensor(np.asarray(observations))
        )
        single = tensor.ndim == 3
        if single:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4 or tuple(tensor.shape[1:]) != self.observation_shape:
            raise ValueError(
                "observations must have shape "
                f"{self.observation_shape} or (N, *{self.observation_shape}), "
                f"got {tuple(tensor.shape)}"
            )
        return (
            to_float_observations(
                tensor.to(device=self.device),
                expected_shape=self.observation_shape,
            ),
            single,
        )

    @staticmethod
    def _validate_epsilon(epsilon: float) -> float:
        value = float(epsilon)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("epsilon must be finite and between 0 and 1")
        return value
