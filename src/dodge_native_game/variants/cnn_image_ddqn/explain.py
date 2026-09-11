"""Offline, gradient-free explanations for the image DDQN network.

The helpers in this module read an already constructed Q-network.  They do
not load checkpoints, step an environment, or alter network parameters.  A
dueling network is exposed as ``Q = V + (A - mean(A))``; a plain Q-head keeps
its direct Q-values and reports no fabricated value stream.
"""

from __future__ import annotations

import operator
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import IMAGE_SHAPE, to_float_observations

CONV_FEATURE_SHAPE: Final = (64, 7, 7)
SHARED_FEATURE_SIZE: Final = 512
DEFAULT_MAX_BATCH_SIZE: Final = 32
DEFAULT_GRID_STRIDE: Final = 12
DEFAULT_BLUR_RADIUS: Final = 6
MAX_GRID_POINTS: Final = 4096

__all__ = [
    "CONV_FEATURE_SHAPE",
    "DEFAULT_BLUR_RADIUS",
    "DEFAULT_GRID_STRIDE",
    "DEFAULT_MAX_BATCH_SIZE",
    "ForwardDecomposition",
    "PairwiseFeatureContributions",
    "ChannelAblation",
    "PerturbationMaps",
    "SHARED_FEATURE_SIZE",
    "channel_ablation",
    "decompose",
    "decompose_forward",
    "explain_forward",
    "explain_observation",
    "forward_decomposition",
    "pairwise_contributions",
    "pairwise_feature_contributions",
    "local_blur",
    "perturbation_maps",
]


@dataclass(frozen=True, slots=True)
class ForwardDecomposition:
    """Intermediate tensors and exact output terms for one network forward.

    ``values`` retains the value head's ``(B, 1)`` shape.  The convenience
    ``value``/``v`` properties squeeze that singleton dimension.  On a plain
    head, ``values``, ``advantages``, and ``centered_advantages`` are all
    ``None`` because the architecture has no such terms.
    """

    q_values: torch.Tensor
    values: torch.Tensor | None
    advantages: torch.Tensor | None
    centered_advantages: torch.Tensor | None
    conv_features: torch.Tensor
    shared_features: torch.Tensor
    dueling: bool
    head_weight: torch.Tensor
    head_bias: torch.Tensor

    @property
    def q(self) -> torch.Tensor:
        return self.q_values

    @property
    def value(self) -> torch.Tensor | None:
        if self.values is None:
            return None
        return self.values.squeeze(-1)

    @property
    def v(self) -> torch.Tensor | None:
        return self.value

    @property
    def advantage(self) -> torch.Tensor | None:
        return self.advantages

    @property
    def centered_advantage(self) -> torch.Tensor | None:
        return self.centered_advantages

    @property
    def centered_a(self) -> torch.Tensor | None:
        return self.centered_advantages

    @property
    def conv(self) -> torch.Tensor:
        return self.conv_features

    @property
    def conv_maps(self) -> torch.Tensor:
        return self.conv_features

    @property
    def features(self) -> torch.Tensor:
        return self.shared_features

    @property
    def head(self) -> str:
        return "dueling" if self.dueling else "plain"

    def to_dict(self, *, include_features: bool = True) -> dict[str, Any]:
        """Return a JSON-compatible copy of the decomposition."""

        result: dict[str, Any] = {
            "q": _json_value(self.q_values),
            "q_values": _json_value(self.q_values),
            "v": _json_value(self.value),
            "value": _json_value(self.value),
            "advantage": _json_value(self.advantages),
            "centered_advantage": _json_value(self.centered_advantages),
            "head": self.head,
        }
        if include_features:
            result.update(
                {
                    "conv_features": _json_value(self.conv_features),
                    "conv_maps": _json_value(self.conv_features),
                    "shared_features": _json_value(self.shared_features),
                }
            )
        return result

    as_dict = to_dict


@dataclass(frozen=True, slots=True)
class PairwiseFeatureContributions:
    """Linear shared-feature terms for one fixed action pair.

    The final column of ``total_contributions`` is the head-bias difference.
    Therefore ``total_contributions.sum(-1)`` reconstructs the chosen-minus-
    alternative Q gap exactly (up to normal floating-point arithmetic).
    """

    chosen: int
    alternative: int
    feature_contributions: torch.Tensor
    bias_difference: torch.Tensor
    total_contributions: torch.Tensor
    q_gap: torch.Tensor
    dueling: bool

    @property
    def contributions(self) -> torch.Tensor:
        """Feature-only terms; use ``total_contributions`` for bias too."""

        return self.feature_contributions

    @property
    def total(self) -> torch.Tensor:
        return self.total_contributions

    @property
    def gap(self) -> torch.Tensor:
        return self.q_gap

    @property
    def bias(self) -> torch.Tensor:
        return self.bias_difference

    def to_dict(self) -> dict[str, Any]:
        return {
            "chosen": self.chosen,
            "alternative": self.alternative,
            "contributions": _json_value(self.feature_contributions),
            "feature_contributions": _json_value(self.feature_contributions),
            "bias_difference": _json_value(self.bias_difference),
            "total_contributions": _json_value(self.total_contributions),
            "q_gap": _json_value(self.q_gap),
            "head": "dueling" if self.dueling else "plain",
        }

    as_dict = to_dict


@dataclass(frozen=True, slots=True)
class ChannelAblation:
    """Outputs after zeroing each selected Conv3 channel end to end."""

    channel_indices: tuple[int, ...]
    original_q: torch.Tensor
    ablated_q: torch.Tensor
    q_delta: torch.Tensor
    original_value: torch.Tensor | None
    ablated_value: torch.Tensor | None
    value_delta: torch.Tensor | None
    chosen: int | None
    alternative: int | None
    original_gap: torch.Tensor | None
    ablated_gap: torch.Tensor | None
    gap_delta: torch.Tensor | None

    @property
    def q(self) -> torch.Tensor:
        return self.ablated_q

    @property
    def delta_q(self) -> torch.Tensor:
        return self.q_delta

    @property
    def channels(self) -> tuple[int, ...]:
        return self.channel_indices

    def to_dict(self) -> dict[str, Any]:
        return {
            "channels": list(self.channel_indices),
            "q": _json_value(self.ablated_q),
            "q_delta": _json_value(self.q_delta),
            "value": _json_value(self.ablated_value),
            "value_delta": _json_value(self.value_delta),
            "chosen": self.chosen,
            "alternative": self.alternative,
            "original_gap": _json_value(self.original_gap),
            "ablated_gap": _json_value(self.ablated_gap),
            "gap_delta": _json_value(self.gap_delta),
        }

    as_dict = to_dict


@dataclass(frozen=True, slots=True)
class PerturbationMaps:
    """Local blur maps for a fixed action pair and stack-channel selection."""

    value_map: torch.Tensor | None
    decision_map: torch.Tensor
    decision_delta_map: torch.Tensor
    original_value: torch.Tensor | None
    perturbed_value: torch.Tensor | None
    original_gap: torch.Tensor
    perturbed_gap: torch.Tensor
    grid_positions: tuple[tuple[int, int], ...]
    grid_shape: tuple[int, int]
    frame: int | None
    channels: tuple[int, ...]
    chosen: int
    alternative: int
    stride: int
    radius: int
    kernel_size: int

    @property
    def value_delta(self) -> torch.Tensor | None:
        return self.value_map

    @property
    def v_delta(self) -> torch.Tensor | None:
        return self.value_map

    @property
    def q_gap(self) -> torch.Tensor:
        return self.perturbed_gap

    @property
    def q_gap_delta(self) -> torch.Tensor:
        return self.decision_delta_map

    def to_dict(self) -> dict[str, Any]:
        return {
            "value_map": _json_value(self.value_map),
            "decision_map": _json_value(self.decision_map),
            "decision_delta_map": _json_value(self.decision_delta_map),
            "original_value": _json_value(self.original_value),
            "perturbed_value": _json_value(self.perturbed_value),
            "original_gap": _json_value(self.original_gap),
            "perturbed_gap": _json_value(self.perturbed_gap),
            "grid_positions": [list(position) for position in self.grid_positions],
            "grid_shape": list(self.grid_shape),
            "frame": self.frame,
            "channels": list(self.channels),
            "chosen": self.chosen,
            "alternative": self.alternative,
            "stride": self.stride,
            "radius": self.radius,
            "kernel_size": self.kernel_size,
        }

    as_dict = to_dict


def decompose(
    model: nn.Module,
    observations: object,
    *,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
) -> ForwardDecomposition:
    """Run a bounded, gradient-free forward decomposition.

    Inputs follow the model boundary: either ``(C,84,84)`` or batched
    ``(B,C,84,84)`` values. Integer frames are normalized identically to the
    model's own forward method. The returned convolution tensor is exactly
    ``(B,64,7,7)`` and the shared tensor exactly ``(B,512)``.
    """

    _require_model(model)
    limit = _positive_int(max_batch_size, "max_batch_size")
    inputs = _prepare_observations(model, observations)

    q_parts: list[torch.Tensor] = []
    conv_parts: list[torch.Tensor] = []
    shared_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    advantage_parts: list[torch.Tensor] = []
    centered_parts: list[torch.Tensor] = []
    head_weight: torch.Tensor | None = None
    head_bias: torch.Tensor | None = None

    with _inference_model(model):
        for start in range(0, inputs.shape[0], limit):
            chunk = inputs[start : start + limit]
            forward = _forward_chunk(model, chunk)
            q, conv, shared, value, advantage, centered, weight, bias = forward
            q_parts.append(q.detach().clone())
            conv_parts.append(conv.detach().clone())
            shared_parts.append(shared.detach().clone())
            if value is not None:
                value_parts.append(value.detach().clone())
                assert advantage is not None
                assert centered is not None
                advantage_parts.append(advantage.detach().clone())
                centered_parts.append(centered.detach().clone())
            if head_weight is None:
                head_weight = weight.detach().clone()
                head_bias = bias.detach().clone()

    if head_weight is None or head_bias is None:
        raise RuntimeError("network produced no head parameters")
    dueling = bool(value_parts)
    return ForwardDecomposition(
        q_values=torch.cat(q_parts, dim=0),
        values=torch.cat(value_parts, dim=0) if dueling else None,
        advantages=torch.cat(advantage_parts, dim=0) if dueling else None,
        centered_advantages=torch.cat(centered_parts, dim=0) if dueling else None,
        conv_features=torch.cat(conv_parts, dim=0),
        shared_features=torch.cat(shared_parts, dim=0),
        dueling=dueling,
        head_weight=head_weight,
        head_bias=head_bias,
    )


decompose_forward = decompose
forward_decomposition = decompose
explain_forward = decompose


def pairwise_feature_contributions(
    source: nn.Module | ForwardDecomposition,
    observations_or_chosen: object | None = None,
    chosen_action: int | None = None,
    alternative_action: int | None = None,
    *,
    chosen: int | None = None,
    alternative: int | None = None,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
) -> PairwiseFeatureContributions:
    """Return exact shared-feature terms for a fixed chosen/alternative pair.

    The first form accepts a model and observations. The second accepts a
    decomposition directly: ``pairwise_feature_contributions(decomposition,
    chosen, alternative)``. For a dueling head, the value and mean-advantage
    terms cancel from the pairwise gap, leaving the advantage-head weight
    difference. A plain head uses its ``q_head`` directly.
    """

    if chosen is not None:
        if chosen_action is not None and chosen_action != chosen:
            raise TypeError("chosen and chosen_action disagree")
        chosen_action = chosen
    if alternative is not None:
        if alternative_action is not None and alternative_action != alternative:
            raise TypeError("alternative and alternative_action disagree")
        alternative_action = alternative

    if isinstance(source, ForwardDecomposition):
        if alternative_action is None and observations_or_chosen is not None:
            if chosen_action is None:
                raise TypeError("alternative action is required")
            chosen_action, alternative_action = (
                observations_or_chosen,
                chosen_action,
            )
        decomposition = source
    else:
        if observations_or_chosen is None:
            raise TypeError("observations are required when source is a model")
        if chosen_action is None or alternative_action is None:
            raise TypeError("chosen and alternative actions are required")
        decomposition = decompose(
            source,
            observations_or_chosen,
            max_batch_size=max_batch_size,
        )

    if chosen_action is None or alternative_action is None:
        raise TypeError("chosen and alternative actions are required")
    chosen_index, alternative_index = _validate_pair(
        chosen_action,
        alternative_action,
        decomposition.q_values.shape[-1],
    )
    weight_delta = (
        decomposition.head_weight[chosen_index]
        - decomposition.head_weight[alternative_index]
    )
    bias_difference = (
        decomposition.head_bias[chosen_index]
        - decomposition.head_bias[alternative_index]
    )
    feature_terms = decomposition.shared_features * weight_delta.unsqueeze(0)
    bias_terms = bias_difference.expand(decomposition.shared_features.shape[0], 1)
    total_terms = torch.cat((feature_terms, bias_terms), dim=1)
    q_gap = decomposition.q_values[:, chosen_index] - decomposition.q_values[
        :, alternative_index
    ]
    return PairwiseFeatureContributions(
        chosen=chosen_index,
        alternative=alternative_index,
        feature_contributions=feature_terms,
        bias_difference=bias_difference,
        total_contributions=total_terms,
        q_gap=q_gap,
        dueling=decomposition.dueling,
    )


pairwise_contributions = pairwise_feature_contributions


def channel_ablation(
    model: nn.Module,
    observations: object,
    chosen_action: int | None = None,
    alternative_action: int | None = None,
    *,
    channels: int | Sequence[int] | None = None,
    channel: int | None = None,
    chosen: int | None = None,
    alternative: int | None = None,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
) -> ChannelAblation:
    """Zero selected input channels and run the complete network each time."""

    _require_model(model)
    if chosen is not None:
        if chosen_action is not None and chosen_action != chosen:
            raise TypeError("chosen and chosen_action disagree")
        chosen_action = chosen
    if alternative is not None:
        if alternative_action is not None and alternative_action != alternative:
            raise TypeError("alternative and alternative_action disagree")
        alternative_action = alternative
    limit = _positive_int(max_batch_size, "max_batch_size")
    inputs = _prepare_observations(model, observations)
    count = CONV_FEATURE_SHAPE[0]
    indices = _resolve_channels(channels, channel, count)
    baseline = decompose(model, inputs, max_batch_size=limit)

    # Repeat in sample-major/channel-minor order. The selected index refers
    # to one of the 64 Conv3 channels, not to an input-stack frame. Each
    # shared+head call still honors max_batch_size.
    ablated_features = baseline.conv_features.repeat_interleave(
        len(indices), dim=0
    ).clone()
    for position, channel_index in enumerate(indices):
        ablated_features[position::len(indices), channel_index] = 0
    ablated = _decompose_from_conv(
        model,
        ablated_features,
        max_batch_size=limit,
    )
    batch_size = inputs.shape[0]
    num_actions = baseline.q_values.shape[-1]
    ablated_q = ablated.q_values.reshape(batch_size, len(indices), num_actions)
    original_q = baseline.q_values
    q_delta = original_q.unsqueeze(1) - ablated_q

    original_value: torch.Tensor | None
    ablated_value: torch.Tensor | None
    value_delta: torch.Tensor | None
    if baseline.values is None or ablated.values is None:
        original_value = None
        ablated_value = None
        value_delta = None
    else:
        original_value = baseline.values.squeeze(-1)
        ablated_value = ablated.values.squeeze(-1).reshape(batch_size, len(indices))
        value_delta = original_value.unsqueeze(1) - ablated_value

    pair: tuple[int, int] | None = None
    if chosen_action is not None or alternative_action is not None:
        if chosen_action is None or alternative_action is None:
            raise TypeError("chosen and alternative actions must be supplied together")
        pair = _validate_pair(chosen_action, alternative_action, num_actions)
    if pair is None:
        original_gap = None
        ablated_gap = None
        gap_delta = None
    else:
        chosen_index, alternative_index = pair
        original_gap = original_q[:, chosen_index] - original_q[:, alternative_index]
        ablated_gap = ablated_q[:, :, chosen_index] - ablated_q[:, :, alternative_index]
        gap_delta = original_gap.unsqueeze(1) - ablated_gap

    return ChannelAblation(
        channel_indices=indices,
        original_q=original_q,
        ablated_q=ablated_q,
        q_delta=q_delta,
        original_value=original_value,
        ablated_value=ablated_value,
        value_delta=value_delta,
        chosen=None if pair is None else pair[0],
        alternative=None if pair is None else pair[1],
        original_gap=original_gap,
        ablated_gap=ablated_gap,
        gap_delta=gap_delta,
    )


def perturbation_maps(
    model: nn.Module,
    observations: object,
    chosen_action: int,
    alternative_action: int,
    *,
    frame: int | str | None = None,
    stride: int = DEFAULT_GRID_STRIDE,
    radius: int = DEFAULT_BLUR_RADIUS,
    kernel_size: int | None = None,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
) -> PerturbationMaps:
    """Map local blur effects using a fixed action pair.

    ``frame=None`` blurs the same local patch in every stack channel;
    supplying a frame index blurs only that channel. ``decision_map`` is the
    signed original-minus-perturbed ``Q[chosen] - Q[alternative]`` for the
    fixed pair. ``value_map`` uses the same signed convention. Every model
    call is bounded by ``max_batch_size``.
    """

    _require_model(model)
    limit = _positive_int(max_batch_size, "max_batch_size")
    stride_value = _positive_int(stride, "stride")
    radius_value = _nonnegative_int(radius, "radius")
    inputs = _prepare_observations(model, observations)
    channels = _resolve_frame(frame, inputs.shape[1])
    frame_index = _frame_index(frame)
    kernel = _blur_kernel_size(radius_value, kernel_size)
    y_positions = _grid_centers(IMAGE_SHAPE[1], stride_value)
    x_positions = _grid_centers(IMAGE_SHAPE[2], stride_value)
    grid_positions = tuple(
        (x, y) for y in y_positions for x in x_positions
    )
    if not grid_positions or len(grid_positions) > MAX_GRID_POINTS:
        raise ValueError(f"perturbation grid must contain 1-{MAX_GRID_POINTS} points")

    baseline = decompose(model, inputs, max_batch_size=limit)
    chosen_index, alternative_index = _validate_pair(
        chosen_action,
        alternative_action,
        baseline.q_values.shape[-1],
    )
    original_gap = baseline.q_values[:, chosen_index] - baseline.q_values[
        :, alternative_index
    ]
    original_value = (
        None if baseline.values is None else baseline.values.squeeze(-1)
    )

    blurred_source = _blur_source(inputs, channels, kernel)
    value_rows: list[torch.Tensor] = []
    perturbed_value_rows: list[torch.Tensor] = []
    perturbed_gap_rows: list[torch.Tensor] = []
    batch_size = inputs.shape[0]
    positions_per_batch = max(1, limit // batch_size)
    for start in range(0, len(grid_positions), positions_per_batch):
        positions = grid_positions[start : start + positions_per_batch]
        perturbed = inputs.repeat(len(positions), 1, 1, 1).clone()
        for position, (x, y) in enumerate(positions):
            y0 = max(0, y - radius_value)
            y1 = min(IMAGE_SHAPE[1], y + radius_value + 1)
            x0 = max(0, x - radius_value)
            x1 = min(IMAGE_SHAPE[2], x + radius_value + 1)
            perturbed[
                position * batch_size : (position + 1) * batch_size,
                channels,
                y0:y1,
                x0:x1,
            ] = blurred_source[:, channels, y0:y1, x0:x1]
        output = decompose(model, perturbed, max_batch_size=limit)
        q_values = output.q_values.reshape(len(positions), batch_size, -1)
        gap = q_values[:, :, chosen_index] - q_values[:, :, alternative_index]
        perturbed_gap_rows.extend(gap.unbind(dim=0))
        if output.values is not None:
            values = output.values.squeeze(-1).reshape(len(positions), batch_size)
            value_rows.extend(values.unbind(dim=0))
            perturbed_value_rows.extend(values.unbind(dim=0))

    grid_shape = (len(y_positions), len(x_positions))
    perturbed_gap = torch.stack(perturbed_gap_rows, dim=1).reshape(
        batch_size, *grid_shape
    )
    decision_map = original_gap[:, None, None] - perturbed_gap
    decision_delta_map = decision_map
    if original_value is None or not value_rows:
        value_map = None
        perturbed_value_map = None
    else:
        perturbed_value_map = torch.stack(perturbed_value_rows, dim=1).reshape(
            batch_size, *grid_shape
        )
        value_map = original_value[:, None, None] - perturbed_value_map

    return PerturbationMaps(
        value_map=value_map,
        decision_map=decision_map,
        decision_delta_map=decision_delta_map,
        original_value=original_value,
        perturbed_value=perturbed_value_map,
        original_gap=original_gap,
        perturbed_gap=perturbed_gap,
        grid_positions=grid_positions,
        grid_shape=grid_shape,
        frame=frame_index,
        channels=channels,
        chosen=chosen_index,
        alternative=alternative_index,
        stride=stride_value,
        radius=radius_value,
        kernel_size=kernel,
    )


local_blur = perturbation_maps


def explain_observation(
    model: nn.Module,
    observation: object,
    *,
    chosen: int,
    alternative: int,
    frame: int | str | None = None,
    channel: int = 0,
    radius: int = DEFAULT_BLUR_RADIUS,
    stride: int = DEFAULT_GRID_STRIDE,
    kernel_size: int | None = None,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
) -> dict[str, Any]:
    """Build the bounded JSON payload used by the offline explanation UI."""

    _require_model(model)
    limit = _positive_int(max_batch_size, "max_batch_size")
    inputs = _prepare_observations(model, observation)
    if inputs.shape[0] != 1:
        raise ValueError("explain_observation expects one CHW observation")
    decomposition = decompose(model, inputs, max_batch_size=limit)
    pair = pairwise_feature_contributions(
        decomposition,
        chosen=chosen,
        alternative=alternative,
        max_batch_size=limit,
    )
    maps = perturbation_maps(
        model,
        inputs,
        chosen,
        alternative,
        frame=frame,
        stride=stride,
        radius=radius,
        kernel_size=kernel_size,
        max_batch_size=limit,
    )
    ablation = channel_ablation(
        model,
        inputs,
        chosen,
        alternative,
        channel=channel,
        max_batch_size=limit,
    )
    ablated_q = ablation.ablated_q[0, 0]
    ablated_gap = (
        None if ablation.gap_delta is None else ablation.gap_delta[0, 0]
    )
    result: dict[str, Any] = {
        "q": _json_value(decomposition.q_values[0]),
        "q_values": _json_value(decomposition.q_values[0]),
        "value": _json_value(
            None if decomposition.value is None else decomposition.value[0]
        ),
        "v": _json_value(
            None if decomposition.value is None else decomposition.value[0]
        ),
        "advantage": _json_value(decomposition.advantages[0])
        if decomposition.advantages is not None
        else None,
        "centered_advantage": _json_value(decomposition.centered_advantages[0])
        if decomposition.centered_advantages is not None
        else None,
        "head": decomposition.head,
        "conv_maps": _json_value(decomposition.conv_features[0]),
        "shared_features": _json_value(decomposition.shared_features[0]),
        "contributions": _json_value(pair.feature_contributions[0]),
        "bias_difference": _json_value(pair.bias_difference),
        "chosen": pair.chosen,
        "alternative": pair.alternative,
        "value_map": _json_value(maps.value_map[0])
        if maps.value_map is not None
        else None,
        "decision_map": _json_value(maps.decision_map[0]),
        "decision_delta_map": _json_value(maps.decision_delta_map[0]),
        "q_gap_map": _json_value(maps.perturbed_gap[0]),
        "q_gap_delta_map": _json_value(maps.decision_delta_map[0]),
        "original_value": _json_value(maps.original_value[0])
        if maps.original_value is not None
        else None,
        "original_gap": _json_value(maps.original_gap[0]),
        "grid_positions": [list(position) for position in maps.grid_positions],
        "grid_shape": list(maps.grid_shape),
        "frame": maps.frame,
        "blur_channels": list(maps.channels),
        "radius": maps.radius,
        "stride": maps.stride,
        "kernel_size": maps.kernel_size,
        "ablation": {
            "channel": channel,
            "q": _json_value(ablated_q),
            "q_delta": _json_value(ablation.q_delta[0, 0]),
            "gap_delta": _json_value(ablated_gap),
            "value": _json_value(ablation.ablated_value[0, 0])
            if ablation.ablated_value is not None
            else None,
            "value_delta": _json_value(ablation.value_delta[0, 0])
            if ablation.value_delta is not None
            else None,
        },
    }
    return result


def _forward_chunk(
    model: nn.Module,
    observations: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
]:
    features_module = getattr(model, "features", None)
    shared_module = getattr(model, "shared", None)
    if not isinstance(features_module, nn.Module) or not isinstance(
        shared_module, nn.Module
    ):
        raise TypeError("model must expose features and shared modules")
    conv = features_module(observations)
    if conv.ndim != 4 or tuple(conv.shape[1:]) != CONV_FEATURE_SHAPE:
        raise ValueError(
            "CNN convolution output must have shape "
            f"(B, {CONV_FEATURE_SHAPE}), got {tuple(conv.shape)}"
        )
    q_values, shared, value, advantage, centered, weight, bias = _forward_from_conv(
        model, conv
    )
    return q_values, conv, shared, value, advantage, centered, weight, bias


def _forward_from_conv(
    model: nn.Module,
    conv: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
]:
    """Run shared and head modules on an already computed Conv3 tensor."""

    if conv.ndim != 4 or tuple(conv.shape[1:]) != CONV_FEATURE_SHAPE:
        raise ValueError(
            "CNN convolution output must have shape "
            f"(B, {CONV_FEATURE_SHAPE}), got {tuple(conv.shape)}"
        )
    shared_module = getattr(model, "shared", None)
    if not isinstance(shared_module, nn.Module):
        raise TypeError("model must expose a shared module")
    shared = shared_module(conv)
    if shared.ndim != 2 or tuple(shared.shape[1:]) != (SHARED_FEATURE_SIZE,):
        raise ValueError(
            "CNN shared output must have shape "
            f"(B, {SHARED_FEATURE_SIZE}), got {tuple(shared.shape)}"
        )

    dueling = bool(getattr(model, "dueling", False))
    if dueling:
        value_stream = getattr(model, "value_stream", None)
        advantage_stream = getattr(model, "advantage_stream", None)
        if not isinstance(value_stream, nn.Linear) or not isinstance(
            advantage_stream, nn.Linear
        ):
            raise TypeError(
                "dueling model must expose value_stream and advantage_stream"
            )
        value = value_stream(shared)
        advantage = advantage_stream(shared)
        centered = advantage - advantage.mean(dim=1, keepdim=True)
        # Match AtariCnnQNetwork.forward's operation order exactly.  Keeping
        # the subtraction outside the first addition avoids a tiny rounding
        # difference in forward-parity checks.
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        weight = advantage_stream.weight
        bias = advantage_stream.bias
    else:
        q_head = getattr(model, "q_head", None)
        if not isinstance(q_head, nn.Linear):
            raise TypeError("plain model must expose q_head")
        value = None
        advantage = None
        centered = None
        q_values = q_head(shared)
        weight = q_head.weight
        bias = q_head.bias
    if bias is None:
        raise TypeError("Q head must have a bias for exact contribution accounting")
    return q_values, shared, value, advantage, centered, weight, bias


def _decompose_from_conv(
    model: nn.Module,
    conv_features: torch.Tensor,
    *,
    max_batch_size: int,
) -> ForwardDecomposition:
    """Decompose shared/head output for Conv3 ablation batches."""

    if conv_features.ndim != 4 or tuple(conv_features.shape[1:]) != CONV_FEATURE_SHAPE:
        raise ValueError(
            "conv_features must have shape "
            f"(B, {CONV_FEATURE_SHAPE}), got {tuple(conv_features.shape)}"
        )
    if conv_features.shape[0] < 1:
        raise ValueError("conv_features batch must not be empty")
    q_parts: list[torch.Tensor] = []
    shared_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    advantage_parts: list[torch.Tensor] = []
    centered_parts: list[torch.Tensor] = []
    head_weight: torch.Tensor | None = None
    head_bias: torch.Tensor | None = None
    with _inference_model(model):
        for start in range(0, conv_features.shape[0], max_batch_size):
            q, shared, value, advantage, centered, weight, bias = _forward_from_conv(
                model,
                conv_features[start : start + max_batch_size],
            )
            q_parts.append(q.detach().clone())
            shared_parts.append(shared.detach().clone())
            if value is not None:
                value_parts.append(value.detach().clone())
                assert advantage is not None
                assert centered is not None
                advantage_parts.append(advantage.detach().clone())
                centered_parts.append(centered.detach().clone())
            if head_weight is None:
                head_weight = weight.detach().clone()
                head_bias = bias.detach().clone()
    if head_weight is None or head_bias is None:
        raise RuntimeError("network produced no head parameters")
    dueling = bool(value_parts)
    return ForwardDecomposition(
        q_values=torch.cat(q_parts, dim=0),
        values=torch.cat(value_parts, dim=0) if dueling else None,
        advantages=torch.cat(advantage_parts, dim=0) if dueling else None,
        centered_advantages=torch.cat(centered_parts, dim=0) if dueling else None,
        conv_features=conv_features.detach().clone(),
        shared_features=torch.cat(shared_parts, dim=0),
        dueling=dueling,
        head_weight=head_weight,
        head_bias=head_bias,
    )


def _prepare_observations(model: nn.Module, observations: object) -> torch.Tensor:
    if isinstance(observations, torch.Tensor):
        tensor = observations
    else:
        try:
            array = np.asarray(observations)
            if not array.flags.writeable:
                array = np.array(array, copy=True)
            tensor = torch.as_tensor(array)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "observations must be array-like or a torch.Tensor"
            ) from exc
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    expected_shape = _observation_shape(model)
    if tensor.ndim != 4 or tuple(tensor.shape[1:]) != expected_shape:
        raise ValueError(
            "observations must have shape "
            f"{expected_shape} or (B, *{expected_shape}), got {tuple(tensor.shape)}"
        )
    if tensor.shape[0] < 1:
        raise ValueError("observations batch must not be empty")
    device = _model_device(model)
    return to_float_observations(
        tensor.to(device=device),
        expected_shape=expected_shape,
    )


def _observation_shape(model: nn.Module) -> tuple[int, int, int]:
    declared = getattr(model, "observation_shape", None)
    if declared is not None:
        try:
            shape = tuple(operator.index(item) for item in declared)
        except (TypeError, ValueError):
            shape = ()
        if len(shape) == 3 and shape[0] > 0 and shape[1:] == (84, 84):
            return shape
        raise ValueError("model observation_shape must be (channels, 84, 84)")
    features = getattr(model, "features", None)
    if isinstance(features, nn.Module):
        for layer in features.modules():
            if isinstance(layer, nn.Conv2d):
                return (int(layer.in_channels), 84, 84)
    raise TypeError("model must declare observation_shape or expose a Conv2d")


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cpu")


@contextmanager
def _inference_model(model: nn.Module) -> Iterator[None]:
    states = [(module, module.training) for module in model.modules()]
    try:
        # Preserve every training flag, including custom nested modules.  The
        # architecture has no stochastic layers, but offline explanations
        # should remain deterministic for compatible networks too.
        for module, _ in states:
            module.training = False
        with torch.inference_mode():
            yield
    finally:
        for module, training in states:
            module.training = training


def _require_model(model: object) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return int(result)


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return int(result)


def _validate_action(value: object, num_actions: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer action index")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer action index") from exc
    if not 0 <= result < num_actions:
        raise ValueError(f"{name} must be between 0 and {num_actions - 1}")
    return int(result)


def _validate_pair(
    chosen: object,
    alternative: object,
    num_actions: int,
) -> tuple[int, int]:
    chosen_index = _validate_action(chosen, num_actions, "chosen")
    alternative_index = _validate_action(alternative, num_actions, "alternative")
    if chosen_index == alternative_index:
        raise ValueError("chosen and alternative actions must differ")
    return chosen_index, alternative_index


def _resolve_channels(
    channels: int | Sequence[int] | None,
    channel: int | None,
    count: int,
) -> tuple[int, ...]:
    if channel is not None:
        if channels is not None:
            raise TypeError("pass either channel or channels, not both")
        channels = channel
    if channels is None:
        raw: Sequence[int] = tuple(range(count))
    elif isinstance(channels, (str, bytes)):
        raise TypeError("channels must be an integer or sequence of integers")
    elif isinstance(channels, Sequence):
        raw = channels
    else:
        raw = (channels,)
    result = tuple(_validate_channel(item, count) for item in raw)
    if not result:
        raise ValueError("at least one channel is required")
    if len(set(result)) != len(result):
        raise ValueError("channels must be unique")
    return result


def _validate_channel(value: object, count: int) -> int:
    if isinstance(value, bool):
        raise TypeError("channel must be an integer index")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError("channel must be an integer index") from exc
    if not 0 <= result < count:
        raise ValueError(f"channel must be between 0 and {count - 1}")
    return int(result)


def _resolve_frame(frame: int | str | None, count: int) -> tuple[int, ...]:
    if frame is None:
        return tuple(range(count))
    if isinstance(frame, str):
        if frame.lower() in {"all", "allstack", "stack", "*"}:
            return tuple(range(count))
        raise ValueError("frame must be an index or 'allstack'")
    return (_validate_channel(frame, count),)


def _frame_index(frame: int | str | None) -> int | None:
    if frame is None:
        return None
    if isinstance(frame, str):
        return None
    return int(operator.index(frame))


def _blur_kernel_size(radius: int, kernel_size: int | None) -> int:
    if kernel_size is None:
        return 2 * radius + 1
    kernel = _positive_int(kernel_size, "kernel_size")
    if kernel % 2 == 0:
        raise ValueError("kernel_size must be odd")
    return kernel


def _grid_centers(length: int, stride: int) -> tuple[int, ...]:
    """Use the center of each stride bin, clamped to the image edge."""

    return tuple(
        min(length - 1, start + stride // 2)
        for start in range(0, length, stride)
    )


def _blur_source(
    inputs: torch.Tensor,
    channels: tuple[int, ...],
    kernel_size: int,
) -> torch.Tensor:
    if kernel_size == 1:
        return inputs
    padding = kernel_size // 2
    selected = inputs[:, channels]
    padded = F.pad(selected, (padding, padding, padding, padding), mode="replicate")
    blurred = F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)
    source = inputs.clone()
    source[:, channels] = blurred
    return source


def _json_value(value: object) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value
