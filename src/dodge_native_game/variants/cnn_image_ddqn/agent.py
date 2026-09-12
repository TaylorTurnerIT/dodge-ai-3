"""Composable Double-DQN learner for the image-only Dodge variant."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import (
    IMAGE_SHAPE,
    AtariCnnQNetwork,
    to_float_observations,
    validate_observation_shape,
)
from .pixels import GRAY_PROFILE, PICO8_LUMA_PALETTE, PICO8_PALETTE, RGB_PROFILE
from .replay import ReplayBatch

if TYPE_CHECKING:
    from .pixel_replay import PackedPixelReplayBatch

NetworkFactory = Callable[[int], nn.Module]
MAX_GRAD_NORM: Final = 10.0
LEARNER_BACKENDS: Final = ("baseline", "cuda-amp", "cuda-optimized")

__all__ = ["DDQNUpdate", "DoubleDQNAgent", "LEARNER_BACKENDS", "MAX_GRAD_NORM"]


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
    diagnostics_sampled: bool = True


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
        learner_backend: str = "baseline",
    ) -> None:
        if isinstance(num_actions, bool) or num_actions < 1:
            raise ValueError("num_actions must be a positive integer")
        if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be finite and between 0 and 1")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        self.observation_shape = validate_observation_shape(observation_shape)
        if online_network is None:
            if network_factory is AtariCnnQNetwork:
                online_network = network_factory(
                    int(num_actions),
                    input_channels=self.observation_shape[0],
                    input_size=self.observation_shape[1],
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
        if learner_backend not in LEARNER_BACKENDS:
            raise ValueError(f"learner_backend must be one of {LEARNER_BACKENDS}")
        if learner_backend != "baseline" and self.device.type != "cuda":
            raise ValueError("CUDA learner backends require a CUDA device")
        self.learner_backend = learner_backend
        self._amp_enabled = learner_backend != "baseline"
        self._non_blocking = learner_backend == "cuda-optimized"
        if learner_backend == "cuda-optimized":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        self.online_network = online_network.to(self.device)
        self.target_network = target_network.to(self.device)
        if learner_backend == "cuda-optimized":
            self.online_network.to(memory_format=torch.channels_last)
            self.target_network.to(memory_format=torch.channels_last)
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=self.learning_rate,
            fused=learner_backend == "cuda-optimized",
        )
        for parameter in self.target_network.parameters():
            parameter.requires_grad_(False)
        self.target_network.eval()
        self._rng = np.random.default_rng(seed)
        self.optimizer_steps = 0
        self._grad_scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled)
        self._online_forward: Callable[[torch.Tensor], torch.Tensor] = (
            self.online_network
        )
        self._target_forward: Callable[[torch.Tensor], torch.Tensor] = (
            self.target_network
        )
        if learner_backend == "cuda-optimized":
            self._online_forward = torch.compile(
                self.online_network, mode="reduce-overhead"
            )
            self._target_forward = torch.compile(
                self.target_network, mode="reduce-overhead"
            )

    def select_action(
        self,
        observations: object,
        epsilon: object,
        *,
        rng: np.random.Generator | None = None,
    ) -> int | np.ndarray:
        """Select greedy or random actions for one or many image observations."""

        tensor, single = self._validated_observation_batch(observations)
        epsilon_values = self._epsilon_values(epsilon, tensor.shape[0])
        random_source = self._rng if rng is None else rng
        random_mask = random_source.random(tensor.shape[0]) < epsilon_values
        actions = np.empty(tensor.shape[0], dtype=np.int64)
        if np.any(random_mask):
            actions[random_mask] = random_source.integers(
                0, self.num_actions, size=int(np.count_nonzero(random_mask))
            )
        greedy_indices = np.flatnonzero(~random_mask)
        if greedy_indices.size:
            selected = (
                tensor
                if greedy_indices.size == tensor.shape[0]
                else tensor[greedy_indices.tolist()]
            )
            selected = to_float_observations(
                self._to_device(selected),
                expected_shape=self.observation_shape,
            )
            with torch.inference_mode():
                q_values = self._checked_q_values(self.online_network, selected)
            actions[greedy_indices] = (
                q_values.argmax(dim=1).cpu().numpy().astype(np.int64)
            )
        if single:
            return int(actions[0])
        return actions

    def update(
        self, batch: ReplayBatch | PackedPixelReplayBatch, *, diagnostics: bool = True
    ) -> DDQNUpdate:
        """Run one Huber-loss Double-DQN update from an explicit batch."""

        from .pixel_replay import PackedPixelReplayBatch

        if not isinstance(batch, (ReplayBatch, PackedPixelReplayBatch)):
            raise TypeError("batch must be a ReplayBatch")
        if batch.observation_shape != self.observation_shape:
            raise ValueError(
                "batch observation shape does not match the agent: "
                f"{batch.observation_shape} != {self.observation_shape}"
            )
        # Validate the small owned CPU action array before transfer. GPU scalar
        # boolean checks here would introduce two synchronization points.
        if np.any(batch.actions < 0) or np.any(batch.actions >= self.num_actions):
            raise ValueError("batch actions contain an out-of-range action")
        if isinstance(batch, PackedPixelReplayBatch):
            observation_profile = getattr(batch, "observation_profile", RGB_PROFILE)
            observations = self._packed_observation_tensor(
                batch.observations, observation_profile=observation_profile
            )
            next_observations = self._packed_observation_tensor(
                batch.next_observations, observation_profile=observation_profile
            )
        else:
            observations, _ = self._observation_tensor(batch.observations)
            next_observations, _ = self._observation_tensor(batch.next_observations)
        actions = self._to_device(torch.from_numpy(batch.actions), dtype=torch.long)
        rewards = self._to_device(torch.from_numpy(batch.rewards), dtype=torch.float32)
        dones = self._to_device(torch.from_numpy(batch.dones), dtype=torch.float32)
        discounts = getattr(batch, "discounts", None)
        if discounts is None:
            bootstrap_discounts: torch.Tensor | float = self.gamma
        else:
            discounts_array = np.asarray(discounts)
            if discounts_array.shape != (batch.size,):
                raise ValueError("batch discounts must have shape (N,)")
            if not np.issubdtype(discounts_array.dtype, np.number):
                raise ValueError("batch discounts must be numeric")
            discounts_array = np.asarray(discounts_array, dtype=np.float32)
            if (
                not np.isfinite(discounts_array).all()
                or np.any(discounts_array < 0.0)
                or np.any(discounts_array > 1.0)
            ):
                raise ValueError("batch discounts must be finite and between 0 and 1")
            bootstrap_discounts = self._to_device(
                torch.from_numpy(discounts_array), dtype=torch.float32
            )
        if actions.ndim != 1 or actions.shape[0] != batch.size:
            raise ValueError("batch actions must have shape (N,)")

        self.online_network.train()
        with self._autocast():
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
                targets = rewards + bootstrap_discounts * (1.0 - dones) * next_q_values

            loss = F.smooth_l1_loss(chosen_q_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        if self._amp_enabled:
            self._grad_scaler.scale(loss).backward()
            self._grad_scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(), max_norm=MAX_GRAD_NORM
        )
        if self._amp_enabled:
            self._grad_scaler.step(self.optimizer)
            self._grad_scaler.update()
        else:
            self.optimizer.step()
        self.optimizer_steps += 1

        if not diagnostics:
            # Unsampled metrics are explicitly missing, never measured zeros.
            return DDQNUpdate(
                loss=float("nan"),
                mean_q=float("nan"),
                mean_target=float("nan"),
                pre_clip_grad_norm=float("nan"),
                batch_size=batch.size,
                optimizer_step=self.optimizer_steps,
                td_error_mean=float("nan"),
                td_error_std=float("nan"),
                target_std=float("nan"),
                q_std=float("nan"),
                diagnostics_sampled=False,
            )

        with torch.no_grad():
            td_errors = targets.detach() - chosen_q_values.detach()
            zero = loss.detach().new_zeros(())
            # One transfer for the entire sampled diagnostic vector, not one
            # synchronization for every statistic on every optimizer update.
            values = (
                torch.stack(
                    (
                        loss.detach(),
                        chosen_q_values.detach().mean(),
                        targets.detach().mean(),
                        pre_clip_grad_norm.detach(),
                        td_errors.mean(),
                        td_errors.std() if batch.size > 1 else zero,
                        targets.detach().std() if batch.size > 1 else zero,
                        chosen_q_values.detach().std() if batch.size > 1 else zero,
                    )
                )
                .cpu()
                .tolist()
            )

        return DDQNUpdate(
            loss=values[0],
            mean_q=values[1],
            mean_target=values[2],
            pre_clip_grad_norm=values[3],
            batch_size=batch.size,
            optimizer_step=self.optimizer_steps,
            td_error_mean=values[4],
            td_error_std=values[5],
            target_std=values[6],
            q_std=values[7],
        )

    def load_optimizer_state_dict(self, state: dict) -> None:
        """Restore Adam state while retaining the selected execution backend."""

        self.optimizer.load_state_dict(state)
        if self.learner_backend == "cuda-optimized":
            for group in self.optimizer.param_groups:
                group["fused"] = True
                group["foreach"] = None
                for parameter in group["params"]:
                    parameter_state = self.optimizer.state.get(parameter, {})
                    for key, value in tuple(parameter_state.items()):
                        if (
                            isinstance(value, torch.Tensor)
                            and value.shape == parameter.shape
                        ):
                            restored = torch.empty_like(
                                parameter, memory_format=torch.preserve_format
                            )
                            restored.copy_(value)
                            parameter_state[key] = restored

    @torch.no_grad()
    def sync_target(self) -> None:
        """Hard-copy online parameters into the target network."""

        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

    @torch.inference_mode()
    def q_values_for(self, observations: object) -> np.ndarray:
        """Return raw Q-values for logging without affecting training."""

        tensor, _ = self._observation_tensor(observations)
        q_values = self._checked_q_values(self.online_network, tensor)
        return q_values.detach().cpu().numpy().astype(np.float64, copy=True)

    @torch.inference_mode()
    def evaluation_values(self, observations: object) -> tuple[np.ndarray, float]:
        """Return Q values and feature sparsity from one online-network forward."""

        q_values, dead_fractions = self.evaluation_batch_values(observations)
        return q_values, float(dead_fractions.mean())

    @torch.inference_mode()
    def evaluation_batch_values(
        self, observations: object
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return Q values and one feature-sparsity value per observation."""

        tensor, _ = self._observation_tensor(observations)
        features = getattr(self.online_network, "features", None)
        captured: list[torch.Tensor] = []
        handle = None
        if isinstance(features, nn.Module):
            handle = features.register_forward_hook(
                lambda _module, _inputs, output: captured.append(output.detach())
            )
        try:
            q_values = self._checked_q_values(self.online_network, tensor)
        finally:
            if handle is not None:
                handle.remove()
        dead_fractions = (
            (captured[-1] == 0.0)
            .to(dtype=torch.float32)
            .flatten(start_dim=1)
            .mean(dim=1)
            if captured and captured[-1].numel()
            else q_values.new_zeros((q_values.shape[0],))
        )
        transferred = torch.cat((q_values.detach().reshape(-1), dead_fractions))
        values = transferred.cpu().numpy().astype(np.float64, copy=True)
        q_size = q_values.numel()
        return values[:q_size].reshape(tuple(q_values.shape)), values[q_size:]

    @torch.inference_mode()
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
        with self._autocast():
            if network is self.online_network:
                q_values = self._online_forward(observations)
            elif network is self.target_network:
                q_values = self._target_forward(observations)
            else:
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
        tensor, single = self._validated_observation_batch(observations)
        return (
            to_float_observations(
                self._to_device(tensor),
                expected_shape=self.observation_shape,
            ),
            single,
        )

    def _validated_observation_batch(
        self, observations: object
    ) -> tuple[torch.Tensor, bool]:
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
        return tensor, single

    def _packed_observation_tensor(
        self,
        observations: np.ndarray,
        *,
        observation_profile: str = RGB_PROFILE,
    ) -> torch.Tensor:
        """Decode packed palette IDs on-device into the agent's input shape."""

        packed_value = np.asarray(observations)
        if packed_value.ndim != 3 or packed_value.dtype != np.uint8:
            raise ValueError("packed observations must have shape (N, stack, bytes)")
        if observation_profile not in (RGB_PROFILE, GRAY_PROFILE):
            raise ValueError(
                "packed observations require profile "
                f"{RGB_PROFILE!r} or {GRAY_PROFILE!r}"
            )
        profile_is_rgb = observation_profile == RGB_PROFILE

        height, width = self.observation_shape[1:]
        if (height, width) != (128, 128):
            raise ValueError("packed palette batches require fixed 128x128 frames")
        expected_channels = (
            3 * int(packed_value.shape[1])
            if profile_is_rgb
            else int(packed_value.shape[1])
        )
        if self.observation_shape[0] != expected_channels:
            raise ValueError(
                "packed observation profile does not match the agent shape: "
                f"{observation_profile!r} -> "
                f"{(expected_channels, height, width)} != "
                f"{self.observation_shape}"
            )
        expected_bytes = height * width // 2
        if height * width % 2 or packed_value.shape[2] != expected_bytes:
            raise ValueError(
                "packed observations do not match the agent spatial shape: "
                f"expected {expected_bytes} bytes per frame, "
                f"got {packed_value.shape[2]}"
            )

        packed = self._to_device(torch.from_numpy(packed_value), dtype=torch.uint8)
        indices = torch.stack((packed >> 4, packed & 15), dim=-1).flatten(-2).long()
        if profile_is_rgb:
            palette = getattr(self, "_display_palette", None)
            if palette is None:
                palette = (
                    torch.tensor(PICO8_PALETTE, dtype=torch.float32, device=self.device)
                    / 255.0
                )
                self._display_palette = palette
            rgb = palette[indices]
            batch_size, stack_size = packed_value.shape[:2]
            result = (
                rgb.reshape(batch_size, stack_size, height, width, 3)
                .permute(0, 1, 4, 2, 3)
                .reshape(batch_size, stack_size * 3, height, width)
            )
            return self._formatted_observations(result)

        luma = getattr(self, "_display_luma_palette", None)
        if luma is None:
            luma = (
                torch.tensor(
                    PICO8_LUMA_PALETTE, dtype=torch.float32, device=self.device
                )
                / 255.0
            )
            self._display_luma_palette = luma
        batch_size, stack_size = packed_value.shape[:2]
        result = luma[indices].reshape(batch_size, stack_size, height, width)
        return self._formatted_observations(result)

    def _to_device(
        self, tensor: torch.Tensor, *, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        value = tensor
        if self._non_blocking and value.device.type == "cpu" and not value.is_pinned():
            value = value.pin_memory()
        value = value.to(
            device=self.device,
            dtype=dtype,
            non_blocking=self._non_blocking,
        )
        return self._formatted_observations(value)

    def _formatted_observations(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.learner_backend == "cuda-optimized" and tensor.ndim == 4:
            return tensor.contiguous(memory_format=torch.channels_last)
        return tensor

    def _autocast(self):
        if not self._amp_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    @staticmethod
    def _validate_epsilon(epsilon: float) -> float:
        value = float(epsilon)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("epsilon must be finite and between 0 and 1")
        return value

    @classmethod
    def _epsilon_values(cls, epsilon: object, batch_size: int) -> np.ndarray:
        values = np.asarray(epsilon)
        if values.ndim == 0:
            return np.full(batch_size, cls._validate_epsilon(float(values)))
        if values.shape != (batch_size,) or not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"epsilon must be scalar or have shape ({batch_size},)")
        values = np.asarray(values, dtype=np.float64)
        if (
            not np.isfinite(values).all()
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise ValueError("epsilon values must be finite and between 0 and 1")
        return values
