"""Matched frozen CLS/projected current-frame decoder study."""

from __future__ import annotations

import base64
import hashlib
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
_TRAIN_FRAMES = 16_384
_VALIDATION_FRAMES = 2_048


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
    decoder_factory: Callable[[int], nn.Module] = QueryPixelDecoder,
) -> tuple[nn.Module, nn.Module]:
    """Create two decoders with byte-identical initial parameters."""

    target = torch.device(device)
    with torch.random.fork_rng(devices=_rng_devices(target)):
        torch.manual_seed(seed)
        first = decoder_factory(latent_dim)
        initial = _state(first)
        second = decoder_factory(latent_dim)
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
    sampler: torch.Generator | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
    on_milestone: Callable[[MatchedSnapshot, nn.Module, nn.Module], None] | None = None,
) -> FitResult:
    """Fit both heads with one shared sampling stream and milestone callback."""

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
                target = (
                    _take(targets, indices)
                    .to(device=device, dtype=torch.float32)
                    .div(255.0)
                )
                prediction = _pixels(decoder(latent), tuple(target.shape))
                loss = F.mse_loss(prediction, target)
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
) -> dict[str, Any]:
    denominator = rows * channels * height * width
    changed_denominator = changed * channels
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
                rows = changed_count = 0
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
                    prediction = decoder(latent)
                    current = torch.from_numpy(
                        np.asarray(pixels[positions], dtype=np.float32) / 255.0  # type: ignore[index]
                    ).to(target_device)
                    prediction = _pixels(prediction, tuple(current.shape))
                    predicted = prediction.detach().cpu().numpy()
                    target = current.detach().cpu().numpy()
                    mean = np.broadcast_to(training_mean, target.shape)
                    mask = np.asarray(changed_masks[positions], dtype=bool)  # type: ignore[index]
                    error = np.square(predicted - target)
                    baseline_error = np.square(mean - target)
                    total += float(error.sum())
                    baseline += float(baseline_error.sum())
                    rows += len(positions)
                    changed_count += int(mask.sum())
                    changed_total += float((error * mask[:, None]).sum())
                    changed_baseline += float((baseline_error * mask[:, None]).sum())
                    for local_index, global_index in enumerate(positions.tolist()):
                        if global_index not in wanted:
                            continue
                        changed_one = mask[local_index]
                        examples[global_index] = {
                            "index": global_index,
                            "split": split,
                            "episode_id": records[global_index]["episode_id"],
                            "frame_index": records[global_index]["frame_index"],
                            "recipe_family": records[global_index].get("recipe_family"),
                            "mse": float(error[local_index].mean()),
                            "training_mean_mse": float(
                                baseline_error[local_index].mean()
                            ),
                            "changed_pixel_count": int(changed_one.sum()),
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
                result["splits"][split] = _stats(
                    rows,
                    total,
                    baseline,
                    changed_count,
                    changed_total,
                    changed_baseline,
                    channels=int(target.shape[1]),
                    height=int(target.shape[2]),
                    width=int(target.shape[3]),
                )
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
) -> Path:
    path = run / f"decoder-{snapshot.step}.pt"
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
) -> list[Path]:
    """Run matched native-resolution CLS and projected decoder fits."""

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
    history = Path(history_root)
    common = {
        "variant": "pixel-repr-ddqn",
        "experiment": _EXPERIMENT,
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
        cls_decoder, projected_decoder = make_decoder_pair(_LATENT_DIM, device=device)
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
                )
                wrong_by_index = {item["index"]: item for item in wrong["examples"]}
                for item in normal["examples"]:
                    paired = wrong_by_index.get(item["index"])
                    if paired is not None:
                        item["wrong_reconstructed"] = paired["reconstructed"]
                clean = {
                    "step": snapshot.step,
                    "representation": mode,
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
                "experiment": _EXPERIMENT,
                "run_id": run_id,
                "diagnostic_only": True,
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
