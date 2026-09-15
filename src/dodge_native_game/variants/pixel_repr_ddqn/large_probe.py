"""Matched frozen CLS/projected current-frame decoder study."""

from __future__ import annotations

import base64
import hashlib
import inspect
import io
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from . import pretrain
from .palette import (
    MAX_PALETTE_SIZE,
    bce_palette_loss,
    ce_palette_loss,
    derive_palette,
    forward_logits,
    palette_indices,
    palette_to_json,
    render_palette_rgb,
    validate_palette_coverage,
)
from .probe_bank import ProbeBank, build_bank
from .query_decoder import QueryPixelDecoder
from .run_artifacts import (
    append_metric,
    atomic_json,
    create_run,
    file_hash,
    write_status,
)

__all__ = [
    "FitResult",
    "FrameBank",
    "MatchedSnapshot",
    "bce_palette_loss",
    "balanced_bright_loss",
    "ce_palette_loss",
    "palette_loss",
    "evaluate_decoder_stream",
    "fit_matched_decoders",
    "make_decoder_pair",
    "open_frame_bank",
    "run_study",
    "stream_train_mean",
]

_EXPERIMENT = "large-current-frame-decoder-v1"
_WORLD_EXPERIMENT = "practice-batch32-v1"
_WORLD_STEP = 512
_WORLD_BATCH_SIZE = 32
_LATENT_DIM = 192
_OUTPUT_SIZE = 128
_INIT_SEED = 904
_SAMPLING_SEED = 903
_DECODER_BATCH_SIZE = 32
_EVAL_BATCH_SIZE = 64
_DECODER_LR = 1e-3
_DECODER_WEIGHT_DECAY = 0.01
_BRIGHT_THRESHOLD = 0.8
_LOSS_NORMALIZATION = "per-frame-then-batch"
_MSE_NORMALIZATION = "global-pixel-mean"
_BALANCED_BRIGHT_EXPERIMENT = "large-current-frame-decoder-balanced-bright-v1"
_PALETTE_CE_EXPERIMENT = "large-current-frame-decoder-palette-ce-v1"
_PALETTE_BCE_EXPERIMENT = "large-current-frame-decoder-palette-bce-v1"
_TRAIN_FRAMES = 16_384
_VALIDATION_FRAMES = 2_048
_PALETTE_BATCH_SIZE = 64
_PALETTE_CE_NORMALIZATION = "unweighted-pixel-mean"
_PALETTE_BCE_NORMALIZATION = "unweighted-pixel-class-mean"


def _rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]


def _clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return value


def _state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in module.state_dict().items()
    }


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _digest(value: object) -> str:
    return hashlib.sha256(
        value if isinstance(value, bytes) else _canonical(value)
    ).hexdigest()


def _validate_loss_kind(loss_kind: str) -> str:
    if loss_kind not in {"mse", "balanced-bright", "palette-ce", "palette-bce"}:
        raise ValueError(
            "loss_kind must be 'mse', 'balanced-bright', 'palette-ce', or "
            "'palette-bce'"
        )
    return loss_kind


def _loss_normalization(loss_kind: str) -> str:
    if loss_kind == "palette-ce":
        return _PALETTE_CE_NORMALIZATION
    if loss_kind == "palette-bce":
        return _PALETTE_BCE_NORMALIZATION
    return _LOSS_NORMALIZATION if loss_kind == "balanced-bright" else _MSE_NORMALIZATION


def _experiment_for_loss(loss_kind: str) -> str:
    return {
        "mse": _EXPERIMENT,
        "balanced-bright": _BALANCED_BRIGHT_EXPERIMENT,
        "palette-ce": _PALETTE_CE_EXPERIMENT,
        "palette-bce": _PALETTE_BCE_EXPERIMENT,
    }[loss_kind]


def _palette_metadata(palette: np.ndarray | None) -> dict[str, Any]:
    """Return one stable palette provenance bundle for every artifact surface."""

    colors = palette_to_json(palette)
    digest = (
        hashlib.sha256(
            np.ascontiguousarray(palette, dtype=np.uint8).tobytes()
        ).hexdigest()
        if palette is not None
        else None
    )
    return {
        "palette_rgb": colors,
        "palette_sha256": digest,
        "palette_source_split": "train" if palette is not None else None,
        "palette_size": len(palette) if palette is not None else None,
        "output_channels": len(palette) if palette is not None else 3,
        "raw_logits": palette is not None,
        "decoder_input_split": "train",
        "train_only_input": True,
    }


def palette_loss(
    logits: torch.Tensor, targets: torch.Tensor, loss_kind: str
) -> torch.Tensor:
    """Apply the selected unweighted per-pixel palette objective."""

    if loss_kind == "palette-ce":
        return ce_palette_loss(logits, targets)
    if loss_kind == "palette-bce":
        return bce_palette_loss(logits, targets)
    raise ValueError("palette_loss requires 'palette-ce' or 'palette-bce'")


def _bright_mask(
    target: torch.Tensor, threshold: float = _BRIGHT_THRESHOLD
) -> torch.Tensor:
    if target.ndim != 4 or target.shape[1] != 3:
        raise ValueError("target must have shape (batch, 3, height, width)")
    return target.amin(dim=1) >= threshold


def balanced_bright_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = _BRIGHT_THRESHOLD,
) -> torch.Tensor:
    """Weight target-bright and target-background pixel classes equally."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    bright = _bright_mask(target, threshold)
    per_pixel = (prediction - target).square().mean(dim=1)
    bright_count = bright.flatten(1).sum(dim=1)
    background_count = (~bright).flatten(1).sum(dim=1)
    bright_mean = (per_pixel * bright).flatten(1).sum(dim=1) / bright_count.clamp_min(1)
    background_mean = (per_pixel * ~bright).flatten(1).sum(
        dim=1
    ) / background_count.clamp_min(1)
    both = (bright_count > 0) & (background_count > 0)
    available = torch.where(bright_count > 0, bright_mean, background_mean)
    per_frame = torch.where(both, 0.5 * bright_mean + 0.5 * background_mean, available)
    return per_frame.mean()


class _Concat:
    """Read-only concatenation of train/validation memmaps."""

    def __init__(self, parts: Sequence[np.ndarray]) -> None:
        self.parts = tuple(parts)
        shape = tuple(self.parts[0].shape[1:])
        if any(tuple(part.shape[1:]) != shape for part in self.parts):
            raise ValueError("bank arrays have inconsistent row shapes")
        self.shape = (sum(int(part.shape[0]) for part in self.parts), *shape)
        self.dtype = self.parts[0].dtype
        self.offsets = np.cumsum([0, *(len(part) for part in self.parts)])

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key: object) -> np.ndarray:
        if isinstance(key, (int, np.integer)):
            index = int(key)
            if index < 0:
                index += len(self)
            for part, start, stop in zip(
                self.parts, self.offsets[:-1], self.offsets[1:], strict=True
            ):
                if start <= index < stop:
                    return np.asarray(part[index - start])
            raise IndexError("bank row out of range")
        positions = np.arange(len(self))[key]
        positions = np.atleast_1d(positions).astype(np.int64, copy=False)
        if positions.size == 0:
            return np.empty((0, *self.shape[1:]), dtype=self.dtype)
        return np.stack([self[int(index)] for index in positions])


@dataclass(frozen=True)
class FrameBank:
    root: Path
    train: ProbeBank
    validation: ProbeBank
    pixels: _Concat
    changed: _Concat
    features: dict[str, _Concat]
    records: tuple[dict[str, Any], ...]
    split_ranges: dict[str, tuple[int, int]]
    data_hash: str
    frame_index_hash: str

    def indices(self, split: str) -> range:
        return range(*self.split_ranges[split])

    @property
    def count(self) -> int:
        return len(self.pixels)


def _recipe_families(dataset_root: Path) -> dict[str, str | None]:
    try:
        payload = json.loads((dataset_root / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    result: dict[str, str | None] = {}
    episodes = payload.get("episodes", {})
    if isinstance(episodes, Mapping):
        for values in episodes.values():
            if isinstance(values, Sequence):
                for value in values:
                    if isinstance(value, Mapping) and value.get("episode_id"):
                        result[str(value["episode_id"])] = value.get("recipe_family")
    return result


def _ensure_bank(
    model: nn.Module,
    dataset_root: Path,
    bank_root: Path,
    *,
    device: str,
    checkpoint_sha256: str,
) -> None:
    bank_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        output = bank_root / split
        if (output / "READY").is_file():
            continue
        build_bank(
            model,
            dataset_root,
            output,
            split,
            device=device,
            frames_per_episode=4,
            seed=_SAMPLING_SEED,
            encode_batch_size=32,
            checkpoint_sha256=checkpoint_sha256,
        )


def open_frame_bank(
    dataset_root: Path,
    bank_root: Path,
    *,
    checkpoint_sha256: str,
) -> FrameBank:
    """Open the concrete train/validation ``ProbeBank`` pair."""

    root = Path(bank_root)
    train, validation = ProbeBank(root / "train"), ProbeBank(root / "validation")
    if (len(train), len(validation)) != (_TRAIN_FRAMES, _VALIDATION_FRAMES):
        raise ValueError("large probe requires 16384 train and 2048 validation frames")
    dataset_hash = file_hash(Path(dataset_root) / "manifest.json")
    for expected_split, bank in (("train", train), ("validation", validation)):
        if bank.metadata.get("split") != expected_split:
            raise ValueError("probe bank split metadata does not match its directory")
        if bank.metadata.get("seed") != _SAMPLING_SEED:
            raise ValueError("probe bank must use the locked sampling seed 903")
        if bank.metadata.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("probe bank checkpoint hash does not match model input")
        if bank.metadata.get("data_hash") != dataset_hash:
            raise ValueError("probe bank data hash does not match dataset manifest")
        if bank.metadata.get("frames_per_episode") != 4:
            raise ValueError("probe bank must contain four frames per episode")
    families = _recipe_families(Path(dataset_root))
    records: list[dict[str, Any]] = []
    for split, bank in (("train", train), ("validation", validation)):
        for index, item in enumerate(bank.index):
            episode_id = str(item["episode_id"])
            records.append(
                {
                    "split": split,
                    "index": index,
                    "episode_id": episode_id,
                    "frame_index": int(item["frame"]),
                    "recipe_family": families.get(episode_id),
                }
            )
    frame_index_hash = _digest({"train": train.index, "validation": validation.index})
    return FrameBank(
        root=root,
        train=train,
        validation=validation,
        pixels=_Concat((train.pixels, validation.pixels)),
        changed=_Concat((train.changed, validation.changed)),
        features={
            "cls": _Concat((train.cls, validation.cls)),
            "projected": _Concat((train.projected, validation.projected)),
        },
        records=tuple(records),
        split_ranges={
            "train": (0, len(train)),
            "validation": (len(train), len(train) + len(validation)),
        },
        data_hash=dataset_hash,
        frame_index_hash=frame_index_hash,
    )


def make_decoder_pair(
    latent_dim: int = _LATENT_DIM,
    *,
    device: torch.device | str = "cpu",
    seed: int = _INIT_SEED,
    output_channels: int = 3,
    raw_logits: bool = False,
    decoder_factory: Callable[..., nn.Module] = QueryPixelDecoder,
) -> tuple[nn.Module, nn.Module]:
    """Create two decoders with byte-identical initial parameters.

    Palette heads use the same ``output_channels`` and raw-logit contract for
    both representations.  Tiny test factories that only accept ``latent_dim``
    retain the historical call shape.
    """

    def create() -> nn.Module:
        if decoder_factory is QueryPixelDecoder:
            return decoder_factory(
                latent_dim,
                output_channels=output_channels,
                raw_logits=raw_logits,
            )
        parameters = inspect.signature(decoder_factory).parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        kwargs = {
            key: value
            for key, value in (
                ("output_channels", output_channels),
                ("raw_logits", raw_logits),
            )
            if key in parameters or accepts_kwargs
        }
        return decoder_factory(latent_dim, **kwargs)

    target = torch.device(device)
    with torch.random.fork_rng(devices=_rng_devices(target)):
        torch.manual_seed(seed)
        first = create()
        initial = _state(first)
        second = create()
        second.load_state_dict(initial, strict=True)
    return first.to(target), second.to(target)


@dataclass(frozen=True)
class MatchedSnapshot:
    step: int
    model: dict[str, dict[str, torch.Tensor]]
    optimizer: dict[str, dict[str, Any]]
    sampler: torch.Tensor
    torch_rng: torch.Tensor
    cuda_rng: list[torch.Tensor] | None


@dataclass(frozen=True)
class FitResult:
    metrics: tuple[dict[str, Any], ...]
    sampled_indices: tuple[tuple[int, ...], ...]
    snapshots: dict[int, MatchedSnapshot]
    sampler: torch.Tensor
    torch_rng: torch.Tensor
    cuda_rng: list[torch.Tensor] | None


def _take(source: object, indices: torch.Tensor) -> torch.Tensor:
    if isinstance(source, torch.Tensor):
        return source.index_select(0, indices.to(source.device))
    return torch.from_numpy(
        np.array(source[indices.cpu().numpy()], copy=True)  # type: ignore[index]
    )


def _pixels(value: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    expected = tuple(shape)
    if value.ndim == 2:
        value = value.reshape(value.shape[0], *expected[1:])
    if tuple(value.shape) != expected:
        raise ValueError(
            f"decoder output must have shape {expected}, got {tuple(value.shape)}"
        )
    return value


def fit_matched_decoders(
    cls_decoder: nn.Module,
    projected_decoder: nn.Module,
    cls_features: object,
    projected_features: object,
    targets: object,
    *,
    milestones: Sequence[int] = (512, 2048, 8192),
    batch_size: int = _DECODER_BATCH_SIZE,
    loss_kind: str = "mse",
    palette: np.ndarray | None = None,
    sampler: torch.Generator | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
    on_milestone: Callable[[MatchedSnapshot, nn.Module, nn.Module], None] | None = None,
) -> FitResult:
    """Fit both heads with one shared sampling stream and milestone callback."""

    loss_kind = _validate_loss_kind(loss_kind)
    palette_loss_kind = loss_kind in {"palette-ce", "palette-bce"}
    if palette_loss_kind and palette is None:
        raise ValueError(f"{loss_kind} requires a train-derived RGB palette")
    if not palette_loss_kind and palette is not None:
        raise ValueError("palette metadata is only valid for palette loss kinds")
    palette_size = len(palette) if palette is not None else None
    schedule = tuple(int(value) for value in milestones)
    if not schedule or tuple(sorted(set(schedule))) != schedule or schedule[0] < 1:
        raise ValueError("milestones must be increasing positive integers")
    if batch_size < 1 or len(targets) < batch_size:  # type: ignore[arg-type]
        raise ValueError("targets must contain one complete decoder batch")
    if len(cls_features) != len(targets) or len(projected_features) != len(targets):  # type: ignore[arg-type]
        raise ValueError("feature and target row counts must match")
    device = next(cls_decoder.parameters()).device
    if next(projected_decoder.parameters()).device != device:
        raise ValueError("matched decoders must share a device")
    cls_decoder.train()
    projected_decoder.train()
    cls_optimizer = torch.optim.AdamW(
        cls_decoder.parameters(), lr=_DECODER_LR, weight_decay=_DECODER_WEIGHT_DECAY
    )
    projected_optimizer = torch.optim.AdamW(
        projected_decoder.parameters(),
        lr=_DECODER_LR,
        weight_decay=_DECODER_WEIGHT_DECAY,
    )
    generator = sampler or torch.Generator(device="cpu").manual_seed(_SAMPLING_SEED)
    metrics: list[dict[str, Any]] = []
    sampled: list[tuple[int, ...]] = []
    snapshots: dict[int, MatchedSnapshot] = {}
    with torch.random.fork_rng(devices=_rng_devices(device)):
        began = time.monotonic()
        for step in range(1, schedule[-1] + 1):
            indices = torch.randint(len(targets), (batch_size,), generator=generator)
            sampled_indices = tuple(int(value) for value in indices.tolist())
            sampled.append(sampled_indices)
            target_rows = _take(targets, indices)
            if palette_loss_kind:
                class_targets = torch.from_numpy(
                    palette_indices(
                        target_rows.detach().cpu().numpy(),  # type: ignore[arg-type]
                        palette,  # type: ignore[arg-type]
                    )
                ).to(device=device, dtype=torch.long)
            else:
                class_targets = None
            losses: dict[str, float] = {}
            for name, decoder, optimizer, features in (
                ("cls", cls_decoder, cls_optimizer, cls_features),
                (
                    "projected",
                    projected_decoder,
                    projected_optimizer,
                    projected_features,
                ),
            ):
                latent = _take(features, indices).to(device=device, dtype=torch.float32)
                if palette_loss_kind:
                    assert class_targets is not None and palette_size is not None
                    expected_shape = (
                        len(indices),
                        palette_size,
                        int(target_rows.shape[-2]),
                        int(target_rows.shape[-1]),
                    )
                    logits = _pixels(decoder(latent), expected_shape)
                    loss = palette_loss(logits, class_targets, loss_kind)
                else:
                    target = target_rows.to(
                        device=device, dtype=torch.float32
                    ).div(255.0)
                    prediction = _pixels(decoder(latent), tuple(target.shape))
                    loss = (
                        F.mse_loss(prediction, target)
                        if loss_kind == "mse"
                        else balanced_bright_loss(prediction, target)
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite decoder loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses[name] = float(loss.detach())
            metric = {
                "step": step,
                "cls_loss": losses["cls"],
                "projected_loss": losses["projected"],
                "loss_kind": loss_kind,
                "bright_threshold": _BRIGHT_THRESHOLD,
                "equal_class_weights": loss_kind == "balanced-bright",
                "palette_size": palette_size,
                "loss_normalization": _loss_normalization(loss_kind),
                "updates_per_second": step / max(time.monotonic() - began, 1e-9),
                "indices": sampled_indices,
            }
            metrics.append(metric)
            if on_step is not None:
                on_step(metric)
            if step in schedule:
                snapshot = MatchedSnapshot(
                    step=step,
                    model={
                        "cls": _state(cls_decoder),
                        "projected": _state(projected_decoder),
                    },
                    optimizer={
                        "cls": _clone(cls_optimizer.state_dict()),
                        "projected": _clone(projected_optimizer.state_dict()),
                    },
                    sampler=generator.get_state().detach().cpu().clone(),
                    torch_rng=torch.get_rng_state().detach().cpu().clone(),
                    cuda_rng=(
                        [
                            state.detach().cpu().clone()
                            for state in torch.cuda.get_rng_state_all()
                        ]
                        if device.type == "cuda"
                        else None
                    ),
                )
                snapshots[step] = snapshot
                if on_milestone is not None:
                    on_milestone(snapshot, cls_decoder, projected_decoder)
        final_torch_rng = torch.get_rng_state().detach().cpu().clone()
        final_cuda_rng = (
            [state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()]
            if device.type == "cuda"
            else None
        )
    cls_decoder.eval()
    projected_decoder.eval()
    return FitResult(
        metrics=tuple(metrics),
        sampled_indices=tuple(sampled),
        snapshots=snapshots,
        sampler=generator.get_state().detach().cpu().clone(),
        torch_rng=final_torch_rng,
        cuda_rng=final_cuda_rng,
    )


def stream_train_mean(
    pixels: object, indices: range, *, batch_size: int = _EVAL_BATCH_SIZE
) -> np.ndarray:
    """Return a normalized train-only mean while streaming memmap rows."""

    if len(indices) == 0:
        raise ValueError("train rows are empty")
    first = np.asarray(pixels[indices.start])  # type: ignore[index]
    if first.ndim != 3:
        raise ValueError("pixels must have shape (rows, channels, height, width)")
    total = np.zeros(first.shape, dtype=np.float64)
    count = 0
    for start in range(indices.start, indices.stop, batch_size):
        rows = np.arange(start, min(start + batch_size, indices.stop))
        values = np.asarray(pixels[rows], dtype=np.float64) / 255.0  # type: ignore[index]
        total += values.sum(axis=0)
        count += len(values)
    return (total / count).astype(np.float32)


def _fixed_examples(records: Sequence[Mapping[str, Any]], rows: range) -> list[int]:
    selected: list[int] = []
    families: set[str] = set()
    for index in rows:
        family = str(records[index].get("recipe_family") or "unknown")
        if family not in families:
            families.add(family)
            selected.append(index)
            if len(selected) == 16:
                return selected
    for index in rows:
        if index not in selected:
            selected.append(index)
            if len(selected) == 16:
                break
    return selected


def _stats(
    rows: int,
    error: float,
    baseline: float,
    changed: int,
    changed_error: float,
    changed_baseline: float,
    *,
    channels: int = 3,
    height: int = _OUTPUT_SIZE,
    width: int = _OUTPUT_SIZE,
    bright_error: float = 0.0,
    background_error: float = 0.0,
    bright_baseline: float = 0.0,
    background_baseline: float = 0.0,
    target_bright: int = 0,
    predicted_bright: int = 0,
    bright_true_positive: int = 0,
    changed_bright_error: float = 0.0,
    changed_bright: int = 0,
) -> dict[str, Any]:
    denominator = rows * channels * height * width
    changed_denominator = changed * channels
    pixels = rows * height * width
    background = pixels - target_bright
    union = target_bright + predicted_bright - bright_true_positive
    return {
        "frame_count": rows,
        "mse": error / denominator if denominator else None,
        "training_mean_mse": baseline / denominator if denominator else None,
        "changed_pixel_count": changed,
        "changed_region_mse": changed_error / changed_denominator if changed else None,
        "changed_region_training_mean_mse": (
            changed_baseline / changed_denominator if changed else None
        ),
        "changed_pixel_fraction": changed / (rows * height * width) if rows else 0.0,
        "bright_threshold": _BRIGHT_THRESHOLD,
        "bright_mse": bright_error / target_bright if target_bright else None,
        "background_mse": (background_error / background if background else None),
        "bright_training_mean_mse": (
            bright_baseline / target_bright if target_bright else None
        ),
        "background_training_mean_mse": (
            background_baseline / background if background else None
        ),
        "training_mean_bright_mse": (
            bright_baseline / target_bright if target_bright else None
        ),
        "training_mean_background_mse": (
            background_baseline / background if background else None
        ),
        "target_bright_share": target_bright / pixels if pixels else 0.0,
        "predicted_bright_share": predicted_bright / pixels if pixels else 0.0,
        "bright_precision": (
            bright_true_positive / predicted_bright if predicted_bright else None
        ),
        "bright_recall": (
            bright_true_positive / target_bright if target_bright else None
        ),
        "bright_iou": bright_true_positive / union if union else None,
        "changed_bright_count": changed_bright,
        "changed_bright_mse": (
            changed_bright_error / changed_bright if changed_bright else None
        ),
    }


def _bright_numpy_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, float | int]:
    target_mask = np.min(target, axis=1) >= _BRIGHT_THRESHOLD
    predicted_mask = np.min(predicted, axis=1) >= _BRIGHT_THRESHOLD
    error = np.square(predicted - target).mean(axis=1)
    baseline_error = np.square(baseline - target).mean(axis=1)
    true_positive = target_mask & predicted_mask
    return {
        "bright_error": float(error[target_mask].sum()),
        "background_error": float(error[~target_mask].sum()),
        "bright_baseline": float(baseline_error[target_mask].sum()),
        "background_baseline": float(baseline_error[~target_mask].sum()),
        "target_bright": int(target_mask.sum()),
        "predicted_bright": int(predicted_mask.sum()),
        "bright_true_positive": int(true_positive.sum()),
    }


def _palette_metrics(
    confusion: np.ndarray,
    changed_confusion: np.ndarray,
) -> dict[str, Any]:
    """Summarize palette confusion, including the changed-pixel subset."""

    target_counts = confusion.sum(axis=1)
    predicted_counts = confusion.sum(axis=0)
    correct = np.diag(confusion)
    changed_target_counts = changed_confusion.sum(axis=1)
    changed_correct = np.diag(changed_confusion)
    total = int(confusion.sum())
    changed_total = int(changed_confusion.sum())
    return {
        "palette_class_count": int(confusion.shape[0]),
        "palette_pixel_accuracy": (
            float(correct.sum() / total) if total else None
        ),
        "palette_target_counts": target_counts.astype(np.int64).tolist(),
        "palette_predicted_counts": predicted_counts.astype(np.int64).tolist(),
        "palette_per_color_recall": [
            float(value / count) if count else None
            for value, count in zip(correct, target_counts, strict=True)
        ],
        "palette_per_color_precision": [
            float(value / count) if count else None
            for value, count in zip(correct, predicted_counts, strict=True)
        ],
        "palette_confusion_matrix": confusion.astype(np.int64).tolist(),
        "palette_changed_pixel_accuracy": (
            float(changed_correct.sum() / changed_total) if changed_total else None
        ),
        "palette_changed_target_counts": (
            changed_target_counts.astype(np.int64).tolist()
        ),
        "palette_changed_predicted_counts": changed_confusion.sum(axis=0)
        .astype(np.int64)
        .tolist(),
        "palette_changed_per_color_recall": [
            float(value / count) if count else None
            for value, count in zip(
                changed_correct, changed_target_counts, strict=True
            )
        ],
        "palette_changed_per_color_precision": [
            float(value / count) if count else None
            for value, count in zip(
                changed_correct, changed_confusion.sum(axis=0), strict=True
            )
        ],
        "palette_changed_confusion_matrix": changed_confusion.astype(np.int64).tolist(),
    }


def evaluate_decoder_stream(
    decoder: nn.Module,
    features: object,
    pixels: object,
    changed_masks: object,
    records: Sequence[Mapping[str, Any]],
    split_ranges: Mapping[str, tuple[int, int]],
    training_mean: np.ndarray,
    *,
    device: torch.device | str,
    batch_size: int = _EVAL_BATCH_SIZE,
    wrong_permutation: bool = False,
    palette: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate all rows in batches; changed masks come from the bank."""

    target_device = torch.device(device)
    modules = list(decoder.modules())
    modes = [module.training for module in modules]
    decoder.eval()
    result: dict[str, Any] = {
        "splits": {},
        "examples": [],
        "wrong_latent": wrong_permutation,
    }
    palette_size = len(palette) if palette is not None else None
    try:
        with (
            torch.random.fork_rng(devices=_rng_devices(target_device)),
            torch.no_grad(),
        ):
            for split in ("train", "validation"):
                start, stop = split_ranges[split]
                wanted = _fixed_examples(records, range(start, stop))
                examples: dict[int, dict[str, Any]] = {}
                total = baseline = changed_total = changed_baseline = 0.0
                bright_error = background_error = 0.0
                bright_baseline = background_baseline = 0.0
                target_bright = predicted_bright = bright_true_positive = 0
                changed_bright_error = 0.0
                changed_bright_count = 0
                rows = changed_count = 0
                palette_confusion = (
                    np.zeros((palette_size, palette_size), dtype=np.int64)
                    if palette_size is not None
                    else None
                )
                palette_changed_confusion = (
                    np.zeros((palette_size, palette_size), dtype=np.int64)
                    if palette_size is not None
                    else None
                )
                for cursor in range(start, stop, batch_size):
                    end = min(cursor + batch_size, stop)
                    positions = np.arange(cursor, end, dtype=np.int64)
                    if wrong_permutation:
                        local = (positions - start + 4) % (stop - start)
                        feature_positions = local + start
                    else:
                        feature_positions = positions
                    latent = torch.from_numpy(
                        np.asarray(features[feature_positions], dtype=np.float32)  # type: ignore[index]
                    ).to(target_device)
                    if palette is not None:
                        logits = forward_logits(decoder, latent)
                    else:
                        logits = decoder(latent)
                    current = torch.from_numpy(
                        np.asarray(pixels[positions], dtype=np.float32) / 255.0  # type: ignore[index]
                    ).to(target_device)
                    if palette is not None:
                        expected_shape = (
                            len(positions),
                            len(palette),
                            int(current.shape[-2]),
                            int(current.shape[-1]),
                        )
                        logits = _pixels(logits, expected_shape)
                        prediction = render_palette_rgb(logits, palette)
                        predicted_classes = (
                            logits.argmax(dim=1).detach().cpu().numpy()
                        )
                    else:
                        prediction = _pixels(logits, tuple(current.shape))
                        predicted_classes = None
                    predicted = prediction.detach().cpu().numpy()
                    target = current.detach().cpu().numpy()
                    mean = np.broadcast_to(training_mean, target.shape)
                    mask = np.asarray(changed_masks[positions], dtype=bool)  # type: ignore[index]
                    error = np.square(predicted - target)
                    baseline_error = np.square(mean - target)
                    bright = _bright_numpy_metrics(predicted, target, mean)
                    bright_error += float(bright["bright_error"])
                    background_error += float(bright["background_error"])
                    bright_baseline += float(bright["bright_baseline"])
                    background_baseline += float(bright["background_baseline"])
                    target_bright += int(bright["target_bright"])
                    predicted_bright += int(bright["predicted_bright"])
                    bright_true_positive += int(bright["bright_true_positive"])
                    total += float(error.sum())
                    baseline += float(baseline_error.sum())
                    rows += len(positions)
                    changed_count += int(mask.sum())
                    changed_total += float((error * mask[:, None]).sum())
                    changed_baseline += float((baseline_error * mask[:, None]).sum())
                    if palette is not None:
                        target_classes = palette_indices(
                            np.asarray(pixels[positions]), palette  # type: ignore[index]
                        )
                        assert palette_confusion is not None
                        assert palette_changed_confusion is not None
                        np.add.at(
                            palette_confusion,
                            (target_classes.reshape(-1), predicted_classes.reshape(-1)),
                            1,
                        )
                        changed_target = target_classes[mask]
                        changed_predicted = predicted_classes[mask]
                        np.add.at(
                            palette_changed_confusion,
                            (changed_target.reshape(-1), changed_predicted.reshape(-1)),
                            1,
                        )
                    target_bright_mask = np.min(target, axis=1) >= _BRIGHT_THRESHOLD
                    changed_bright_mask = mask & target_bright_mask
                    changed_bright_count += int(changed_bright_mask.sum())
                    changed_bright_error += float(
                        error.mean(axis=1)[changed_bright_mask].sum()
                    )
                    for local_index, global_index in enumerate(positions.tolist()):
                        if global_index not in wanted:
                            continue
                        changed_one = mask[local_index]
                        changed_bright_one = (
                            changed_one & target_bright_mask[local_index]
                        )
                        one_bright = _bright_numpy_metrics(
                            predicted[local_index : local_index + 1],
                            target[local_index : local_index + 1],
                            mean[local_index : local_index + 1],
                        )
                        one_pixels = target.shape[2] * target.shape[3]
                        one_background = one_pixels - int(one_bright["target_bright"])
                        one_union = (
                            int(one_bright["target_bright"])
                            + int(one_bright["predicted_bright"])
                            - int(one_bright["bright_true_positive"])
                        )
                        example = {
                            "index": global_index,
                            "split": split,
                            "episode_id": records[global_index]["episode_id"],
                            "frame_index": records[global_index]["frame_index"],
                            "recipe_family": records[global_index].get("recipe_family"),
                            "mse": float(error[local_index].mean()),
                            "training_mean_mse": float(
                                baseline_error[local_index].mean()
                            ),
                            "bright_mse": (
                                float(one_bright["bright_error"])
                                / int(one_bright["target_bright"])
                                if one_bright["target_bright"]
                                else None
                            ),
                            "background_mse": (
                                float(one_bright["background_error"]) / one_background
                                if one_background
                                else None
                            ),
                            "bright_training_mean_mse": (
                                float(one_bright["bright_baseline"])
                                / int(one_bright["target_bright"])
                                if one_bright["target_bright"]
                                else None
                            ),
                            "background_training_mean_mse": (
                                float(one_bright["background_baseline"])
                                / one_background
                                if one_background
                                else None
                            ),
                            "target_bright_share": int(one_bright["target_bright"])
                            / one_pixels,
                            "predicted_bright_share": int(
                                one_bright["predicted_bright"]
                            )
                            / one_pixels,
                            "bright_precision": (
                                int(one_bright["bright_true_positive"])
                                / int(one_bright["predicted_bright"])
                                if one_bright["predicted_bright"]
                                else None
                            ),
                            "bright_recall": (
                                int(one_bright["bright_true_positive"])
                                / int(one_bright["target_bright"])
                                if one_bright["target_bright"]
                                else None
                            ),
                            "bright_iou": (
                                int(one_bright["bright_true_positive"]) / one_union
                                if one_union
                                else None
                            ),
                            "changed_pixel_count": int(changed_one.sum()),
                            "changed_bright_count": int(changed_bright_one.sum()),
                            "changed_bright_mse": (
                                float(
                                    error[local_index]
                                    .mean(axis=0)[changed_bright_one]
                                    .mean()
                                )
                                if bool(changed_bright_one.any())
                                else None
                            ),
                            "changed_region_mse": (
                                float(error[local_index][:, changed_one].mean())
                                if bool(changed_one.any())
                                else None
                            ),
                            "changed_region_training_mean_mse": (
                                float(
                                    baseline_error[local_index][:, changed_one].mean()
                                )
                                if bool(changed_one.any())
                                else None
                            ),
                            "observed": np.asarray(pixels[global_index]).copy(),  # type: ignore[index]
                            "reconstructed": np.asarray(predicted[local_index]).copy(),
                            "training_mean": np.asarray(training_mean).copy(),
                        }
                        if palette is not None:
                            example["palette_pixel_accuracy"] = float(
                                (
                                    predicted_classes[local_index]
                                    == target_classes[local_index]
                                ).mean()
                            )
                            if bool(changed_one.any()):
                                example["palette_changed_pixel_accuracy"] = float(
                                    (
                                        predicted_classes[local_index][changed_one]
                                        == target_classes[local_index][changed_one]
                                    ).mean()
                                )
                            else:
                                example["palette_changed_pixel_accuracy"] = None
                        examples[global_index] = example
                split_stats = _stats(
                    rows,
                    total,
                    baseline,
                    changed_count,
                    changed_total,
                    changed_baseline,
                    channels=int(target.shape[1]),
                    height=int(target.shape[2]),
                    width=int(target.shape[3]),
                    bright_error=bright_error,
                    background_error=background_error,
                    bright_baseline=bright_baseline,
                    background_baseline=background_baseline,
                    target_bright=target_bright,
                    predicted_bright=predicted_bright,
                    bright_true_positive=bright_true_positive,
                    changed_bright_error=changed_bright_error,
                    changed_bright=changed_bright_count,
                )
                if palette is not None:
                    assert palette_confusion is not None
                    assert palette_changed_confusion is not None
                    split_stats.update(
                        _palette_metrics(palette_confusion, palette_changed_confusion)
                    )
                result["splits"][split] = split_stats
                result["examples"].extend(
                    examples[index] for index in wanted if index in examples
                )
    finally:
        for module, mode in zip(modules, modes, strict=True):
            module.training = mode
    return result


def _png_bytes(frame: np.ndarray) -> bytes:
    values = np.asarray(frame)
    if values.dtype != np.uint8:
        values = np.clip(values * 255.0, 0, 255).round().astype(np.uint8)
    if values.shape[0] == 3:
        values = values.transpose(1, 2, 0)
    stream = io.BytesIO()
    Image.fromarray(values, mode="RGB").save(stream, format="PNG")
    return stream.getvalue()


def _save_png(path: Path, frame: np.ndarray) -> None:
    _atomic_bytes(path, _png_bytes(frame))


def _data_url(frame: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(_png_bytes(frame)).decode()


def _clean_examples(examples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in example.items()
            if key
            not in {
                "observed",
                "reconstructed",
                "training_mean",
                "wrong_latent",
                "wrong_reconstructed",
            }
        }
        for example in examples
    ]


def _save_decoder_checkpoint(
    run: Path,
    snapshot: MatchedSnapshot,
    mode: str,
    *,
    world_hash: str,
    data_hash: str,
    frame_index_hash: str,
    final_step: int,
    loss_kind: str,
    palette: np.ndarray | None = None,
) -> Path:
    path = run / f"decoder-{snapshot.step}.pt"
    palette_metadata = _palette_metadata(palette)
    pretrain.save_checkpoint(
        path,
        {
            "model": snapshot.model[mode],
            "optimizer": snapshot.optimizer[mode],
            "sampler": snapshot.sampler,
            "sampling_rng": snapshot.sampler,
            "torch_rng": snapshot.torch_rng,
            "cuda_rng": snapshot.cuda_rng,
            "step": snapshot.step,
            "decoder_kind": mode,
            "latent_dim": _LATENT_DIM,
            "decoder_seed": _INIT_SEED,
            "sampling_seed": _SAMPLING_SEED,
            "experiment": _experiment_for_loss(loss_kind),
            "loss_kind": loss_kind,
            "bright_threshold": _BRIGHT_THRESHOLD,
            "equal_class_weights": loss_kind == "balanced-bright",
            "loss_class_weights": {"bright": 0.5, "background": 0.5},
            "loss_normalization": _loss_normalization(loss_kind),
            **palette_metadata,
            "world_model_sha256": world_hash,
            "data_sha256": data_hash,
            "frame_index_sha256": frame_index_hash,
            "diagnostic_only": True,
        },
    )
    if snapshot.step == final_step:
        _link_or_copy(path, run / "decoder.pt")
    return path


def _validate_protocol(
    payload: Mapping[str, Any], model: nn.Module, device: str
) -> str:
    if payload.get("inference_only") is not True:
        raise ValueError("large probe requires a frozen calibrated checkpoint")
    if (
        payload.get("experiment") != _WORLD_EXPERIMENT
        or payload.get("step") != _WORLD_STEP
    ):
        raise ValueError(
            "large probe requires the calibrated batch32 step512 checkpoint"
        )
    if (
        payload.get("batch_size") != _WORLD_BATCH_SIZE
        or payload.get("profile") != "reference"
    ):
        raise ValueError("large probe requires the reference batch32 checkpoint")
    if not isinstance(payload.get("calibration"), Mapping):
        raise ValueError("large probe requires an encoder-calibrated checkpoint")
    if getattr(getattr(model, "config", None), "embed_dim", None) != _LATENT_DIM:
        raise ValueError("large probe requires latent dimension 192")
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("large probe requires CUDA")
    gpu = torch.cuda.get_device_name(0)
    if "T4" not in gpu:
        raise RuntimeError(f"large probe requires an actual T4, got {gpu}")
    return gpu


def _assert_frozen(
    checkpoint: Path, digest: str, model: nn.Module, before: Mapping[str, torch.Tensor]
) -> None:
    if file_hash(checkpoint) != digest:
        raise RuntimeError("world-model checkpoint changed during large probe")
    for key, value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(), before[key]):
            raise RuntimeError(f"frozen world model changed: {key}")


def _write_visuals(
    run: Path,
    evaluation: Mapping[str, Any],
    *,
    mode: str,
    step: int,
    world_hash: str,
    decoder_hash: str,
    palette: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    root = run / "images" / f"step-{step}"
    root.mkdir(parents=True, exist_ok=True)
    mean = None
    all_views: list[dict[str, Any]] = []
    for ordinal, example in enumerate(evaluation["examples"]):
        split = str(example["split"])
        split_ordinal = sum(
            item["split"] == split for item in evaluation["examples"][:ordinal]
        )
        observed = example["observed"]
        reconstruction = example["reconstructed"]
        train_mean = example["training_mean"]
        wrong = example.get("wrong_reconstructed", reconstruction)
        if mean is None:
            mean = train_mean
            _save_png(root / "training-mean.png", train_mean)
        prefix = f"{split}-{split_ordinal:02d}"
        observed_path = root / f"{prefix}-observed.png"
        reconstruction_path = root / f"{prefix}-reconstructed.png"
        wrong_path = root / f"{prefix}-wrong-latent.png"
        _save_png(observed_path, observed)
        _save_png(reconstruction_path, reconstruction)
        _save_png(wrong_path, wrong)
        all_views.append(
            {
                "step": step,
                "model_label": (
                    f"Frozen current-frame {mode.upper()} decoder — diagnostic only"
                ),
                "diagnostic_only": True,
                "frames": [
                    {"label": "Observed current", "image": _data_url(observed)},
                    {
                        "label": "Reconstructed current (diagnostic)",
                        "image": _data_url(reconstruction),
                    },
                    {
                        "label": "Training mean current baseline",
                        "image": _data_url(train_mean),
                    },
                    {
                        "label": "Wrong-latent permutation control",
                        "image": _data_url(wrong),
                    },
                ],
                "features": {
                    "values": [],
                    "label": f"{mode.upper()} latent coordinates; not spatial",
                },
                "metadata": {
                    "current_frame_only": True,
                    "label": (
                        f"{split} {example['episode_id']} frame{example['frame_index']}"
                    ),
                    "split": split,
                    "episode_id": example["episode_id"],
                    "frame": example["frame_index"],
                    "decoder_steps": step,
                    "representation": mode,
                    "checkpoint_sha256": world_hash,
                    "decoder_sha256": decoder_hash,
                    "observed_path": str(observed_path),
                    "reconstructed_path": str(reconstruction_path),
                    "wrong_latent_path": str(wrong_path),
                    **_palette_metadata(palette),
                },
            }
        )
    train_views = [view for view in all_views if view["metadata"]["split"] == "train"]
    validation_views = [
        view for view in all_views if view["metadata"]["split"] == "validation"
    ]
    return train_views[::2][:8] + validation_views[::2][:8]


def run_study(
    checkpoint: Path,
    dataset_root: Path,
    history_root: Path,
    run_id: str,
    *,
    milestones: Sequence[int] = (512, 2048, 8192),
    batch_size: int = _DECODER_BATCH_SIZE,
    device: str = "cuda",
    bank_root: Path | None = None,
    loss_kind: str = "mse",
) -> list[Path]:
    """Run matched native-resolution CLS and projected decoder fits."""

    loss_kind = _validate_loss_kind(loss_kind)
    schedule = tuple(int(value) for value in milestones)
    if not schedule or tuple(sorted(set(schedule))) != schedule or schedule[0] < 1:
        raise ValueError("milestones must be increasing positive integers")
    if batch_size != _DECODER_BATCH_SIZE:
        raise ValueError("large probe requires batch32")
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "checkpoint.pt"
    checkpoint_sha256 = file_hash(checkpoint_path)
    model, payload = pretrain.load_model(checkpoint_path)
    gpu = _validate_protocol(payload, model, device)
    model = model.to(device).eval().requires_grad_(False)
    before = _state(model)
    shared_bank = (
        Path(bank_root)
        if bank_root is not None
        else Path(tempfile.gettempdir())
        / f"dodge-pixel-probe-bank-{checkpoint_sha256[:12]}"
    )
    _ensure_bank(
        model,
        Path(dataset_root),
        shared_bank,
        device=device,
        checkpoint_sha256=checkpoint_sha256,
    )
    bank = open_frame_bank(
        Path(dataset_root), shared_bank, checkpoint_sha256=checkpoint_sha256
    )
    palette = None
    if loss_kind in {"palette-ce", "palette-bce"}:
        palette = derive_palette(
            bank.train.pixels,
            range(len(bank.train.pixels)),
            batch_size=_PALETTE_BATCH_SIZE,
            max_colors=MAX_PALETTE_SIZE,
        )
        validate_palette_coverage(
            bank.validation.pixels,
            range(len(bank.validation.pixels)),
            palette,
            batch_size=_PALETTE_BATCH_SIZE,
            split="validation",
        )
    history = Path(history_root)
    experiment = _experiment_for_loss(loss_kind)
    palette_metadata = _palette_metadata(palette)
    common = {
        "variant": "pixel-repr-ddqn",
        "experiment": experiment,
        "profile": "reference",
        "model_label": "Frozen LeWM current-frame decoder diagnostic",
        "current_frame_only": True,
        "world_model_sha256": checkpoint_sha256,
        "world_model_data_sha256": payload.get("data_hash"),
        "data_hash": bank.data_hash,
        "frame_index_sha256": bank.frame_index_hash,
        "bank_root": str(shared_bank.resolve()),
        "dataset_root": str(Path(dataset_root).resolve()),
        "bank_counts": {"train": _TRAIN_FRAMES, "validation": _VALIDATION_FRAMES},
        "latent_dim": _LATENT_DIM,
        "output_size": _OUTPUT_SIZE,
        "batch_size": batch_size,
        "evaluation_batch_size": _EVAL_BATCH_SIZE,
        "milestones": list(schedule),
        "decoder_init_seed": _INIT_SEED,
        "sampling_seed": _SAMPLING_SEED,
        "loss_kind": loss_kind,
        "bright_threshold": _BRIGHT_THRESHOLD,
        "equal_class_weights": loss_kind == "balanced-bright",
        "loss_class_weights": {"bright": 0.5, "background": 0.5},
        "loss_normalization": _loss_normalization(loss_kind),
        "optimizer": {
            "name": "AdamW",
            "lr": _DECODER_LR,
            "weight_decay": _DECODER_WEIGHT_DECAY,
        },
        "device": device,
        "gpu": gpu,
        "diagnostic_only": True,
        "world_model_fits": False,
        "predictor_used": False,
        **palette_metadata,
    }
    runs: dict[str, Path] = {}
    evaluations: dict[str, dict[int, dict[str, Any]]] = {"cls": {}, "projected": {}}
    try:
        for mode in ("cls", "projected"):
            runs[mode] = create_run(
                history,
                f"{run_id}-{mode}",
                common
                | {
                    "representation": mode,
                    "model_label": (
                        f"Frozen LeWM {mode.upper()} current-frame decoder diagnostic"
                    ),
                },
            )
            _link_or_copy(checkpoint_path, runs[mode] / "checkpoint.pt")
            atomic_json(runs[mode] / "config.json", common | {"representation": mode})
            write_status(
                runs[mode],
                state="running",
                phase="decoder fitting",
                step=0,
                total_steps=schedule[-1],
                message="Frozen world model; decoder only",
            )
        training_mean = stream_train_mean(bank.pixels, bank.indices("train"))
        for path in runs.values():
            _save_png(path / "training-mean.png", training_mean)
        cls_decoder, projected_decoder = make_decoder_pair(
            _LATENT_DIM,
            device=device,
            output_channels=len(palette) if palette is not None else 3,
            raw_logits=palette is not None,
        )
        sampler = torch.Generator(device="cpu").manual_seed(_SAMPLING_SEED)

        def on_step(metric: dict[str, Any]) -> None:
            step = int(metric["step"])
            if step % 128 == 0 or step == schedule[-1]:
                print(
                    json.dumps(
                        {
                            "run_id": run_id,
                            **{
                                key: value
                                for key, value in metric.items()
                                if key != "indices"
                            },
                        }
                    ),
                    flush=True,
                )
                for mode in runs:
                    append_metric(
                        runs[mode] / "metrics.jsonl",
                        {
                            "step": step,
                            "representation": mode,
                            "loss": metric[f"{mode}_loss"],
                            "updates_per_second": metric["updates_per_second"],
                            "phase": "frozen current-frame decoder fitting",
                        },
                    )
                    write_status(
                        runs[mode],
                        state="running",
                        phase="decoder fitting",
                        step=step,
                        total_steps=schedule[-1],
                        message="World model frozen; decoder only",
                    )

        def on_milestone(
            snapshot: MatchedSnapshot, first: nn.Module, second: nn.Module
        ) -> None:
            for mode, decoder in (("cls", first), ("projected", second)):
                decoder_path = _save_decoder_checkpoint(
                    runs[mode],
                    snapshot,
                    mode,
                    world_hash=checkpoint_sha256,
                    data_hash=bank.data_hash,
                    frame_index_hash=bank.frame_index_hash,
                    final_step=schedule[-1],
                    loss_kind=loss_kind,
                    palette=palette,
                )
                normal = evaluate_decoder_stream(
                    decoder,
                    bank.features[mode],
                    bank.pixels,
                    bank.changed,
                    bank.records,
                    bank.split_ranges,
                    training_mean,
                    device=device,
                    batch_size=_EVAL_BATCH_SIZE,
                    palette=palette,
                )
                wrong = evaluate_decoder_stream(
                    decoder,
                    bank.features[mode],
                    bank.pixels,
                    bank.changed,
                    bank.records,
                    bank.split_ranges,
                    training_mean,
                    device=device,
                    batch_size=_EVAL_BATCH_SIZE,
                    wrong_permutation=True,
                    palette=palette,
                )
                wrong_by_index = {item["index"]: item for item in wrong["examples"]}
                for item in normal["examples"]:
                    paired = wrong_by_index.get(item["index"])
                    if paired is not None:
                        item["wrong_reconstructed"] = paired["reconstructed"]
                clean = {
                    "step": snapshot.step,
                    "representation": mode,
                    "loss_kind": loss_kind,
                    "bright_threshold": _BRIGHT_THRESHOLD,
                    "equal_class_weights": loss_kind == "balanced-bright",
                    "loss_normalization": _loss_normalization(loss_kind),
                    **palette_metadata,
                    "world_model_sha256": checkpoint_sha256,
                    "data_sha256": bank.data_hash,
                    "frame_index_sha256": bank.frame_index_hash,
                    "splits": normal["splits"],
                    "wrong_latent_control": wrong["splits"],
                    "examples": _clean_examples(normal["examples"]),
                    "evaluation_only": True,
                }
                atomic_json(runs[mode] / f"evaluation-{snapshot.step}.json", clean)
                decoder_hash = file_hash(decoder_path)
                views = _write_visuals(
                    runs[mode],
                    normal,
                    mode=mode,
                    step=snapshot.step,
                    world_hash=checkpoint_sha256,
                    decoder_hash=decoder_hash,
                    palette=palette,
                )
                atomic_json(runs[mode] / f"visualizations-{snapshot.step}.json", views)
                atomic_json(runs[mode] / "visualizations.json", views)
                validation_stats = normal["splits"]["validation"]
                train_stats = normal["splits"]["train"]
                append_metric(
                    runs[mode] / "metrics.jsonl",
                    {
                        "step": snapshot.step,
                        "representation": mode,
                        "loss": None,
                        "loss_kind": loss_kind,
                        "bright_threshold": _BRIGHT_THRESHOLD,
                        "loss_normalization": _loss_normalization(loss_kind),
                        "train_mse": train_stats["mse"],
                        "validation_mse": validation_stats["mse"],
                        "validation_changed_mse": validation_stats[
                            "changed_region_mse"
                        ],
                        "trainmean_validation_mse": validation_stats[
                            "training_mean_mse"
                        ],
                        "training_mean_validation_mse": validation_stats[
                            "training_mean_mse"
                        ],
                        "validation_changed_trainmean_mse": validation_stats[
                            "changed_region_training_mean_mse"
                        ],
                        "validation_bright_mse": validation_stats["bright_mse"],
                        "validation_background_mse": validation_stats["background_mse"],
                        "validation_bright_training_mean_mse": validation_stats[
                            "bright_training_mean_mse"
                        ],
                        "validation_background_training_mean_mse": validation_stats[
                            "background_training_mean_mse"
                        ],
                        "validation_target_bright_share": validation_stats[
                            "target_bright_share"
                        ],
                        "validation_predicted_bright_share": validation_stats[
                            "predicted_bright_share"
                        ],
                        "validation_bright_precision": validation_stats[
                            "bright_precision"
                        ],
                        "validation_bright_recall": validation_stats["bright_recall"],
                        "validation_bright_iou": validation_stats["bright_iou"],
                        "validation_changed_bright_mse": validation_stats[
                            "changed_bright_mse"
                        ],
                        "validation_changed_bright_count": validation_stats[
                            "changed_bright_count"
                        ],
                        **{
                            f"validation_{key}": value
                            for key, value in validation_stats.items()
                            if key.startswith("palette_")
                        },
                        "phase": "frozen current-frame decoder evaluation",
                    },
                )
                evaluations[mode][snapshot.step] = clean
                write_status(
                    runs[mode],
                    state="completed" if snapshot.step == schedule[-1] else "running",
                    phase="decoder evaluation",
                    step=snapshot.step,
                    total_steps=schedule[-1],
                    message="Milestone evaluated; world model unchanged",
                )

        fit_matched_decoders(
            cls_decoder,
            projected_decoder,
            bank.train.cls,
            bank.train.projected,
            bank.train.pixels,
            milestones=schedule,
            batch_size=batch_size,
            loss_kind=loss_kind,
            palette=palette,
            sampler=sampler,
            on_step=on_step,
            on_milestone=on_milestone,
        )
        _assert_frozen(checkpoint_path, checkpoint_sha256, model, before)
        for mode in runs:
            atomic_json(
                runs[mode] / "report.json",
                {
                    "quality_gate": "inference-only diagnostic",
                    "diagnostic_only": True,
                    "representation": mode,
                    "loss_kind": loss_kind,
                    "bright_threshold": _BRIGHT_THRESHOLD,
                    "equal_class_weights": loss_kind == "balanced-bright",
                    "loss_normalization": _loss_normalization(loss_kind),
                    **palette_metadata,
                    "world_model_sha256": checkpoint_sha256,
                    "data_sha256": bank.data_hash,
                    "frame_index_sha256": bank.frame_index_hash,
                    "milestones": list(schedule),
                    "evaluation": evaluations[mode],
                    "limitations": [
                        "current-frame native128 pixel reconstruction only",
                        "world-model weights and buffers stayed frozen",
                        "predictor and future frames were unused",
                        "changed-region metrics use bank changed masks and "
                        "return null when empty",
                    ],
                },
            )
        atomic_json(
            history / f"{run_id}-comparison.json",
            {
                "variant": "pixel-repr-ddqn",
                "experiment": experiment,
                "run_id": run_id,
                "diagnostic_only": True,
                "loss_kind": loss_kind,
                "bright_threshold": _BRIGHT_THRESHOLD,
                "equal_class_weights": loss_kind == "balanced-bright",
                "loss_normalization": _loss_normalization(loss_kind),
                **palette_metadata,
                "checkpoint_sha256": checkpoint_sha256,
                "data_sha256": bank.data_hash,
                "frame_index_sha256": bank.frame_index_hash,
                "milestones": list(schedule),
                "runs": {mode: str(path) for mode, path in runs.items()},
                "world_model_fits": False,
                "predictor_used": False,
            },
        )
        return [runs["cls"], runs["projected"]]
    except BaseException as error:
        for path in runs.values():
            write_status(
                path,
                state="failed",
                phase="large decoder probe",
                step=0,
                total_steps=schedule[-1],
                message=str(error),
            )
        raise
