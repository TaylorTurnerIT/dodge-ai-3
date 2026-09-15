"""Matched raw-logit palette decoders for CLS, patch-token, and pixel probes.

The fitter owns only the diagnostic head optimization.  Feature extraction,
train/validation bank construction, and the T4 launch policy stay with the
caller.  One CPU sampler supplies every head's minibatch so the three
conditions remain directly paired.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

import numpy as np
import torch
from torch import nn

from .large_probe import FitResult, MatchedSnapshot, _clone, _rng_devices, _state
from .palette import (
    MAX_PALETTE_SIZE,
    ce_palette_loss,
    forward_logits,
    palette_indices,
    palette_to_json,
)
from .spatial_readout import LocalPatchDecoder

__all__ = [
    "CONDITIONS",
    "FEATURE_DIM",
    "MAX_STEPS",
    "NUM_PATCH_TOKENS",
    "OUTPUT_SIZE",
    "SPATIAL_BATCH_SIZE",
    "SPATIAL_DECODER_LR",
    "SPATIAL_DECODER_WEIGHT_DECAY",
    "fit_spatial_decoders",
    "make_spatial_decoders",
]

CONDITIONS: Final[tuple[str, str, str]] = ("cls", "patch", "pixels")
FEATURE_DIM: Final[int] = 192
NUM_PATCH_TOKENS: Final[int] = 256
OUTPUT_SIZE: Final[int] = 128
PALETTE_CLASSES: Final[int] = 3
MAX_STEPS: Final[int] = 8192
SPATIAL_BATCH_SIZE: Final[int] = 32
SPATIAL_DECODER_LR: Final[float] = 1e-3
SPATIAL_DECODER_WEIGHT_DECAY: Final[float] = 0.01
INIT_SEED: Final[int] = 904
SAMPLING_SEED: Final[int] = 903
_MAX_ROWS: Final[int] = 1_000_000


def _decoder_has_dropout(decoder: nn.Module) -> bool:
    dropout_types = (
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
        nn.FeatureAlphaDropout,
    )
    return any(isinstance(module, dropout_types) for module in decoder.modules())


def _states_equal(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    if tuple(left) != tuple(right):
        return False
    return all(torch.equal(left[name], right[name]) for name in left)


def make_spatial_decoders(
    device: torch.device | str = "cpu", seed: int = INIT_SEED
) -> dict[str, nn.Module]:
    """Create three byte-identical raw-logit ``LocalPatchDecoder`` heads."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    target = torch.device(device)
    with torch.random.fork_rng(devices=_rng_devices(target)):
        torch.manual_seed(seed)
        decoders = {
            condition: LocalPatchDecoder().to(target) for condition in CONDITIONS
        }
        initial = _state(decoders[CONDITIONS[0]])
        for condition in CONDITIONS[1:]:
            if not _states_equal(initial, _state(decoders[condition])):
                decoders[condition].load_state_dict(initial, strict=True)
            if not _states_equal(initial, _state(decoders[condition])):
                raise RuntimeError("spatial decoders did not share initialization")
    if any(_decoder_has_dropout(decoder) for decoder in decoders.values()):
        raise ValueError("spatial decoders must not contain dropout")
    return decoders


def _take(source: object, indices: torch.Tensor) -> torch.Tensor:
    if isinstance(source, torch.Tensor):
        return source.index_select(0, indices.to(source.device))
    try:
        values = source[indices.cpu().numpy()]  # type: ignore[index]
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "feature or target source does not support row indexing"
        ) from error
    if isinstance(values, torch.Tensor):
        return values
    try:
        return torch.from_numpy(np.array(values, copy=True))
    except (TypeError, ValueError) as error:
        raise ValueError("feature or target source rows must be numeric") from error


def _source_length(source: object, *, name: str) -> int:
    try:
        count = len(source)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{name} must be a row-indexable source") from error
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise TypeError(f"{name} length must be an integer")
    count = int(count)
    if count < 1:
        raise ValueError(f"{name} must not be empty")
    if count > _MAX_ROWS:
        raise ValueError(f"{name} exceeds the {_MAX_ROWS}-row cap")
    return count


def _validate_palette(palette: object) -> np.ndarray:
    try:
        values = palette_to_json(palette)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError("palette must be a sorted uint8 RGB palette") from error
    if values is None:
        raise ValueError("palette is required for spatial palette decoding")
    colors = np.asarray(values, dtype=np.uint8)
    if len(colors) != PALETTE_CLASSES:
        raise ValueError(
            f"spatial palette decoding requires exactly {PALETTE_CLASSES} colors"
        )
    if len(colors) > MAX_PALETTE_SIZE:
        raise ValueError(f"palette exceeds cap {MAX_PALETTE_SIZE}")
    return colors


def _validate_schedule(
    milestones: Sequence[int], *, batch_size: int, row_count: int
) -> tuple[int, ...]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size < 1 or batch_size > SPATIAL_BATCH_SIZE:
        raise ValueError(f"batch_size must be in 1..{SPATIAL_BATCH_SIZE}")
    if row_count < batch_size:
        raise ValueError("targets must contain one complete decoder batch")
    if isinstance(milestones, (str, bytes)):
        raise TypeError("milestones must be increasing positive integers")
    values: list[int] = []
    for value in milestones:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("milestones must be increasing positive integers")
        values.append(int(value))
    schedule = tuple(values)
    if (
        not schedule
        or tuple(sorted(set(schedule))) != schedule
        or schedule[0] < 1
        or schedule[-1] > MAX_STEPS
    ):
        raise ValueError(
            f"milestones must be increasing positive integers capped at {MAX_STEPS}"
        )
    return schedule


def _validate_feature_batch(values: torch.Tensor, *, name: str) -> torch.Tensor:
    if values.ndim != 3 or tuple(values.shape[1:]) != (
        NUM_PATCH_TOKENS,
        FEATURE_DIM,
    ):
        raise ValueError(
            f"{name} features must have shape (batch, {NUM_PATCH_TOKENS}, "
            f"{FEATURE_DIM}), got {tuple(values.shape)}"
        )
    if not values.is_floating_point():
        raise TypeError(f"{name} features must be floating point")
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} features contain nonfinite values")
    return values


def _validate_target_batch(values: torch.Tensor) -> torch.Tensor:
    expected = (3, OUTPUT_SIZE, OUTPUT_SIZE)
    if values.ndim != 4 or tuple(values.shape[1:]) != expected:
        raise ValueError(
            f"targets must have shape (batch, 3, {OUTPUT_SIZE}, {OUTPUT_SIZE}), "
            f"got {tuple(values.shape)}"
        )
    if values.dtype != torch.uint8:
        raise TypeError("targets must have dtype uint8")
    return values


def _validate_inputs(
    decoders: Mapping[str, nn.Module],
    features: Mapping[str, object],
    targets: object,
    palette: object,
    *,
    batch_size: int,
    milestones: Sequence[int],
) -> tuple[np.ndarray, int, torch.device, tuple[int, ...]]:
    if set(decoders) != set(CONDITIONS):
        raise ValueError(f"decoders must contain exactly {CONDITIONS}")
    if set(features) != set(CONDITIONS):
        raise ValueError(f"features must contain exactly {CONDITIONS}")
    row_count = _source_length(targets, name="targets")
    for condition in CONDITIONS:
        if (
            _source_length(features[condition], name=f"{condition} features")
            != row_count
        ):
            raise ValueError(
                "all feature sources and targets must have equal row counts"
            )
    schedule = _validate_schedule(
        milestones, batch_size=batch_size, row_count=row_count
    )
    colors = _validate_palette(palette)
    devices = {
        next(decoder.parameters()).device for decoder in decoders.values()
    }
    if len(devices) != 1:
        raise ValueError("matched spatial decoders must share a device")
    device = next(iter(devices))
    initial = _state(decoders[CONDITIONS[0]])
    for condition in CONDITIONS[1:]:
        if not _states_equal(initial, _state(decoders[condition])):
            raise ValueError("matched spatial decoders must share initialization")
    if any(_decoder_has_dropout(decoder) for decoder in decoders.values()):
        raise ValueError("spatial decoders must not contain dropout")
    probe_indices = torch.zeros(1, dtype=torch.long)
    _validate_target_batch(_take(targets, probe_indices))
    for condition in CONDITIONS:
        _validate_feature_batch(
            _take(features[condition], probe_indices), name=condition
        )
    return colors, row_count, device, schedule


def _gradient_norm(decoder: nn.Module) -> torch.Tensor:
    squared = torch.zeros((), device=next(decoder.parameters()).device)
    for parameter in decoder.parameters():
        if parameter.grad is not None:
            squared = squared + parameter.grad.detach().square().sum()
    return squared.sqrt()


def _snapshot(
    *,
    step: int,
    decoders: Mapping[str, nn.Module],
    optimizers: Mapping[str, torch.optim.Optimizer],
    sampler: torch.Generator,
    device: torch.device,
) -> MatchedSnapshot:
    return MatchedSnapshot(
        step=step,
        model={condition: _state(decoders[condition]) for condition in CONDITIONS},
        optimizer={
            condition: _clone(optimizers[condition].state_dict())
            for condition in CONDITIONS
        },
        sampler=sampler.get_state().detach().cpu().clone(),
        torch_rng=torch.get_rng_state().detach().cpu().clone(),
        cuda_rng=(
            [state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()]
            if device.type == "cuda"
            else None
        ),
    )


def fit_spatial_decoders(
    decoders: Mapping[str, nn.Module],
    features: Mapping[str, object],
    targets: object,
    palette: object,
    *,
    milestones: Sequence[int] = (512, 2048, 8192),
    batch_size: int = SPATIAL_BATCH_SIZE,
    on_step: Callable[[dict[str, Any]], None] | None = None,
    on_milestone: Callable[[MatchedSnapshot, Mapping[str, nn.Module]], None]
    | None = None,
) -> FitResult:
    """Fit matched CE heads from one shared train-only sampling stream."""

    colors, row_count, device, schedule = _validate_inputs(
        decoders,
        features,
        targets,
        palette,
        batch_size=batch_size,
        milestones=milestones,
    )
    optimizers = {
        condition: torch.optim.AdamW(
            decoders[condition].parameters(),
            lr=SPATIAL_DECODER_LR,
            weight_decay=SPATIAL_DECODER_WEIGHT_DECAY,
        )
        for condition in CONDITIONS
    }
    sampler = torch.Generator(device="cpu").manual_seed(SAMPLING_SEED)
    metrics: list[dict[str, Any]] = []
    sampled_indices: list[tuple[int, ...]] = []
    snapshots: dict[int, MatchedSnapshot] = {}
    original_modes = {
        condition: decoders[condition].training for condition in CONDITIONS
    }
    with torch.random.fork_rng(devices=_rng_devices(device)):
        began = time.monotonic()
        for decoder in decoders.values():
            decoder.train()
        try:
            for step in range(1, schedule[-1] + 1):
                indices = torch.randint(
                    row_count, (batch_size,), generator=sampler, device="cpu"
                )
                paired_indices = tuple(int(value) for value in indices.tolist())
                sampled_indices.append(paired_indices)
                target_rows = _validate_target_batch(_take(targets, indices))
                class_targets = torch.from_numpy(
                    palette_indices(target_rows.detach().cpu().numpy(), colors)
                ).to(device=device, dtype=torch.long)
                losses: dict[str, float] = {}
                for condition in CONDITIONS:
                    decoder = decoders[condition]
                    latent = _validate_feature_batch(
                        _take(features[condition], indices), name=condition
                    ).to(device=device, dtype=torch.float32)
                    optimizer = optimizers[condition]
                    optimizer.zero_grad(set_to_none=True)
                    logits = forward_logits(decoder, latent)
                    expected = (
                        batch_size,
                        len(colors),
                        OUTPUT_SIZE,
                        OUTPUT_SIZE,
                    )
                    if not isinstance(logits, torch.Tensor) or tuple(
                        logits.shape
                    ) != expected:
                        raise ValueError(
                            f"{condition} decoder must return raw logits with shape "
                            f"{expected}, got {getattr(logits, 'shape', None)}"
                        )
                    loss = ce_palette_loss(logits, class_targets)
                    if not torch.isfinite(loss):
                        raise RuntimeError("nonfinite spatial decoder loss")
                    loss.backward()
                    grad_norm = _gradient_norm(decoder)
                    if not torch.isfinite(grad_norm):
                        raise RuntimeError("nonfinite spatial decoder gradient")
                    optimizer.step()
                    losses[condition] = float(loss.detach().cpu())
                metric = {
                    "step": step,
                    "cls_loss": losses["cls"],
                    "patch_loss": losses["patch"],
                    "pixels_loss": losses["pixels"],
                    "loss_kind": "palette-ce",
                    "palette_size": len(colors),
                    "loss_normalization": "unweighted-pixel-mean",
                    "updates_per_second": step / max(time.monotonic() - began, 1e-9),
                    "indices": paired_indices,
                }
                metrics.append(metric)
                if on_step is not None:
                    on_step(metric)
                if step in schedule:
                    snapshot = _snapshot(
                        step=step,
                        decoders=decoders,
                        optimizers=optimizers,
                        sampler=sampler,
                        device=device,
                    )
                    snapshots[step] = snapshot
                    if on_milestone is not None:
                        on_milestone(snapshot, decoders)
                    for decoder in decoders.values():
                        decoder.train()
            final_torch_rng = torch.get_rng_state().detach().cpu().clone()
            final_cuda_rng = (
                [
                    state.detach().cpu().clone()
                    for state in torch.cuda.get_rng_state_all()
                ]
                if device.type == "cuda"
                else None
            )
        finally:
            for condition in CONDITIONS:
                decoders[condition].train(original_modes[condition])
    return FitResult(
        metrics=tuple(metrics),
        sampled_indices=tuple(sampled_indices),
        snapshots=snapshots,
        sampler=sampler.get_state().detach().cpu().clone(),
        torch_rng=final_torch_rng,
        cuda_rng=final_cuda_rng,
    )
