"""Paired RGB and palette-input LeWM training for the §Y screen.

This module deliberately owns a separate, fixed training protocol.  The
legacy MVP/practice runner remains unchanged: both arms receive the same
native uint8 batches, while the model configuration selects the input
encoding.  A single sampling stream and restored stochastic RNG state make
the two updates a matched pair.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn

from .large_dataset import (
    LARGE_DEFAULT_CACHE_SIZE,
    LARGE_DEFAULT_HISTORY_SIZE,
    LargePixelSequenceDataset,
    _read_episode,
    read_dataset_metadata,
)
from .model import (
    INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC,
    INPUT_ENCODING_RGB_NEAREST_SYMMETRIC,
    LeWMConfig,
    LeWorldModel,
)
from .pretrain import save_checkpoint
from .probe_bank import _select_frames
from .run_artifacts import (
    append_metric,
    atomic_json,
    create_run,
    file_hash,
    write_status,
)

__all__ = [
    "ARMS",
    "BATCH_SIZE",
    "CHECKPOINT_STEPS",
    "EXPERIMENT",
    "INIT_SEED",
    "PALETTE_FRAMES_PER_EPISODE",
    "PALETTE_SELECTION_SEED",
    "STEPS",
    "STOCHASTIC_SEED",
    "SAMPLING_SEED",
    "PaletteProvenance",
    "SharedBatch",
    "continue_run",
    "discover_palette",
    "run_pair",
    "sample_shared_batch",
]

EXPERIMENT: Final[str] = "lewm-input-encoding-v1"
ARMS: Final[tuple[str, str]] = ("rgb", "palette")
STEPS: Final[int] = 1024
BATCH_SIZE: Final[int] = 32
CHECKPOINT_STEPS: Final[tuple[int, int]] = (512, 1024)
INIT_SEED: Final[int] = 42
SAMPLING_SEED: Final[int] = 43
STOCHASTIC_SEED: Final[int] = 44
PALETTE_SELECTION_SEED: Final[int] = 903
PALETTE_FRAMES_PER_EPISODE: Final[int] = 4
LEARNING_RATE: Final[float] = 5e-5
WEIGHT_DECAY: Final[float] = 1e-3
GRADIENT_CLIP: Final[float] = 1.0


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _pretty_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _palette_hash(colors: Sequence[Sequence[int]]) -> str:
    packed = bytes(channel for color in colors for channel in color)
    return _sha256_bytes(packed)


def _index_hash(index: Sequence[Mapping[str, object]]) -> str:
    """Match ``probe_bank``'s ``atomic_json(index.json)`` hash exactly."""

    return _sha256_bytes(_pretty_json(list(index)))


@dataclass(frozen=True, slots=True)
class PaletteProvenance:
    """Train-only palette identity and the deterministic frame selection."""

    palette_rgb: tuple[tuple[int, int, int], ...]
    palette_sha256: str
    dataset_manifest_sha256: str
    frame_index_sha256: str
    selected_pixels_sha256: str
    frame_count: int
    frames_per_episode: int
    sampling_seed: int
    source_split: str = "train"

    def as_dict(self) -> dict[str, object]:
        return {
            "palette_rgb": [list(color) for color in self.palette_rgb],
            "palette_sha256": self.palette_sha256,
            "palette_source_split": self.source_split,
            "palette_dataset_manifest_sha256": self.dataset_manifest_sha256,
            "palette_frame_index_sha256": self.frame_index_sha256,
            "palette_selected_pixels_sha256": self.selected_pixels_sha256,
            "palette_frame_count": self.frame_count,
            "palette_frames_per_episode": self.frames_per_episode,
            "palette_sampling_seed": self.sampling_seed,
        }


@dataclass(frozen=True, slots=True)
class SharedBatch:
    """One native minibatch sampled once for both input arms."""

    indices: tuple[int, ...]
    episode_ids: tuple[str, ...]
    starts: tuple[int, ...]
    pixels: torch.Tensor
    actions: torch.Tensor

    def trace_entry(self, step: int) -> dict[str, object]:
        return {
            "step": step,
            "indices": list(self.indices),
            "episode_ids": list(self.episode_ids),
            "starts": list(self.starts),
        }


@dataclass(frozen=True, slots=True)
class _RNGState:
    cpu: torch.Tensor
    cuda: tuple[torch.Tensor, ...] | None


def _capture_rng(device: str) -> _RNGState:
    cuda = None
    if device.startswith("cuda"):
        cuda = tuple(value.clone() for value in torch.cuda.get_rng_state_all())
    return _RNGState(torch.get_rng_state().clone(), cuda)


def _restore_rng(state: _RNGState) -> None:
    torch.set_rng_state(state.cpu)
    if state.cuda is not None:
        torch.cuda.set_rng_state_all(list(state.cuda))


def _rng_digest(state: _RNGState) -> str:
    digest = hashlib.sha256()
    digest.update(state.cpu.numpy().tobytes())
    if state.cuda is not None:
        for value in state.cuda:
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _same_rng(left: _RNGState, right: _RNGState) -> bool:
    if not torch.equal(left.cpu, right.cpu):
        return False
    if left.cuda is None or right.cuda is None:
        return left.cuda is right.cuda
    return len(left.cuda) == len(right.cuda) and all(
        torch.equal(a, b) for a, b in zip(left.cuda, right.cuda, strict=True)
    )


def _trace_hash(entries: Sequence[Mapping[str, object]]) -> str:
    return _sha256_bytes(_canonical_json(list(entries)))


def _canonical_config(config: LeWMConfig) -> dict[str, object]:
    """Convert tuple-valued palette config into JSON/load_model values."""

    return json.loads(json.dumps(asdict(config), allow_nan=False))


def _selected_pixels_hash_update(digest: hashlib._Hash, pixels: np.ndarray) -> None:
    digest.update(np.ascontiguousarray(pixels, dtype=np.uint8).tobytes())


def discover_palette(dataset_root: Path) -> PaletteProvenance:
    """Derive the K=3 input palette from the locked train frame selection.

    The selection is the same four frames per episode, seed 903 selection used
    by the probe bank.  Validation episode arrays are never opened here.
    """

    root = Path(dataset_root)
    metadata = read_dataset_metadata(root)
    dataset_hash = file_hash(root / "manifest.json")
    selector = np.random.default_rng(PALETTE_SELECTION_SEED)
    selected_index: list[dict[str, object]] = []
    observed: set[int] = set()
    selected_pixels = hashlib.sha256()
    for record in metadata.records["train"]:
        frames = _select_frames(
            record.count,
            PALETTE_FRAMES_PER_EPISODE,
            selector,
        )
        pixels, _ = _read_episode(record, verify_hash=True)
        for frame in frames:
            frame_index = int(frame)
            native = pixels[frame_index]
            _selected_pixels_hash_update(selected_pixels, native)
            packed = (
                (native[0].astype(np.uint32) << 16)
                | (native[1].astype(np.uint32) << 8)
                | native[2].astype(np.uint32)
            )
            observed.update(int(value) for value in np.unique(packed))
            selected_index.append(
                {"episode_id": record.episode_id, "frame": frame_index}
            )
            if len(observed) > 3:
                raise ValueError(
                    "input palette must contain exactly three train RGB colors"
                )
        del pixels
    if len(observed) != 3:
        raise ValueError(
            "input palette must contain exactly three train RGB colors, "
            f"got {len(observed)}"
        )
    colors = tuple(
        (
            (value >> 16) & 0xFF,
            (value >> 8) & 0xFF,
            value & 0xFF,
        )
        for value in sorted(observed)
    )
    return PaletteProvenance(
        palette_rgb=colors,
        palette_sha256=_palette_hash(colors),
        dataset_manifest_sha256=dataset_hash,
        frame_index_sha256=_index_hash(selected_index),
        selected_pixels_sha256=selected_pixels.hexdigest(),
        frame_count=len(selected_index),
        frames_per_episode=PALETTE_FRAMES_PER_EPISODE,
        sampling_seed=PALETTE_SELECTION_SEED,
    )


def sample_shared_batch(
    dataset: LargePixelSequenceDataset,
    size: int,
    rng: torch.Generator,
) -> SharedBatch:
    """Sample and materialize one native batch for both world-model arms."""

    if dataset.split != "train":
        raise ValueError("paired world training requires the train split")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("batch size must be a positive integer")
    indices = tuple(
        int(value)
        for value in torch.randint(len(dataset), (size,), generator=rng).tolist()
    )
    samples = [dataset[index] for index in indices]
    try:
        pixels = torch.stack([sample["pixels"] for sample in samples])
        actions = torch.stack([sample["actions"] for sample in samples])
        episode_ids = tuple(str(sample["episode_id"]) for sample in samples)
        starts = tuple(int(sample["start"]) for sample in samples)
    except (KeyError, TypeError, RuntimeError) as error:
        raise ValueError(
            "large train dataset returned an invalid training window"
        ) from error
    if pixels.dtype != torch.uint8 or actions.dtype != torch.int64:
        raise ValueError(
            "paired training requires native uint8 pixels and int64 actions"
        )
    return SharedBatch(indices, episode_ids, starts, pixels, actions)


def _model_state_equal(left: nn.Module, right: nn.Module) -> bool:
    left_state, right_state = left.state_dict(), right.state_dict()
    if tuple(left_state) != tuple(right_state):
        return False
    return all(
        torch.equal(left_state[name], right_state[name]) for name in left_state
    )


def _model_state_sha256(model: nn.Module) -> str:
    """Digest one model state for the matched-initialization audit."""

    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(_canonical_json(list(tensor.shape)))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _parameter_count(model: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def _build_models(
    palette: PaletteProvenance,
    *,
    device: str,
) -> dict[str, LeWorldModel]:
    colors = palette.palette_rgb
    configs = {
        "rgb": LeWMConfig.reference(
            input_encoding=INPUT_ENCODING_RGB_NEAREST_SYMMETRIC,
        ),
        "palette": LeWMConfig.reference(
            input_encoding=INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC,
            palette_rgb=colors,
        ),
    }
    models: dict[str, LeWorldModel] = {}
    for arm in ARMS:
        torch.manual_seed(INIT_SEED)
        models[arm] = LeWorldModel(configs[arm])
    if not _model_state_equal(models["rgb"], models["palette"]):
        raise RuntimeError(
            "RGB and palette arms did not receive matched initial weights"
        )
    return {arm: models[arm].to(device) for arm in ARMS}


def _train_arm_step(
    model: LeWorldModel,
    optimizer: torch.optim.Optimizer,
    pixels: torch.Tensor,
    actions: torch.Tensor,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    losses = model.compute_loss(pixels, actions)
    if not torch.isfinite(losses["loss"]):
        raise RuntimeError("nonfinite paired LeWM training loss")
    losses["loss"].backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
    if not torch.isfinite(grad):
        raise RuntimeError("nonfinite paired LeWM gradient norm")
    optimizer.step()
    values = {
        "loss": float(losses["loss"].detach().cpu()),
        "pred_loss": float(losses["pred_loss"].detach().cpu()),
        "sigreg_loss": float(losses["sigreg_loss"].detach().cpu()),
        "grad_norm": float(grad.detach().cpu()),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError("nonfinite paired LeWM metric")
    return values


def _paired_update(
    models: Mapping[str, LeWorldModel],
    optimizers: Mapping[str, torch.optim.Optimizer],
    batch: SharedBatch,
    *,
    stochastic_state: _RNGState,
    device: str,
) -> tuple[dict[str, dict[str, float]], dict[str, _RNGState]]:
    """Run both arms from one pre-step stochastic state."""

    pixels = batch.pixels.to(device)
    actions = batch.actions.to(device)
    metrics: dict[str, dict[str, float]] = {}
    post_states: dict[str, _RNGState] = {}
    for arm in ARMS:
        _restore_rng(stochastic_state)
        model = models[arm]
        model.train()
        metrics[arm] = _train_arm_step(model, optimizers[arm], pixels, actions)
        post_states[arm] = _capture_rng(device)
    if not _same_rng(post_states["rgb"], post_states["palette"]):
        raise RuntimeError("paired arms consumed different stochastic RNG streams")
    return metrics, post_states


def _validate_run_protocol(
    *,
    steps: int,
    batch_size: int,
    device: str,
    threads: int,
) -> None:
    if steps != STEPS:
        raise ValueError(f"{EXPERIMENT} requires exactly {STEPS} updates")
    if batch_size != BATCH_SIZE:
        raise ValueError(f"{EXPERIMENT} requires batch{BATCH_SIZE}")
    if device != "cuda":
        raise ValueError(f"{EXPERIMENT} requires CUDA T4; CPU fitting is not permitted")
    if threads < 1:
        raise ValueError("threads must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required: run paired input training on a T4")
    if "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("paired input training requires an NVIDIA T4")


def _arm_metadata(
    *,
    run_id: str,
    arm: str,
    config: LeWMConfig,
    config_payload: dict[str, object],
    palette: PaletteProvenance,
    data_hash: str,
    pair_run_id: str,
    initial_state_sha256: str,
    parameter_count: int,
) -> dict[str, object]:
    return {
        "variant": "pixel-repr-ddqn",
        "experiment": EXPERIMENT,
        "run_id": run_id,
        "pair_run_id": pair_run_id,
        "input_arm": arm,
        "input_encoding": config.input_encoding,
        "config": config_payload,
        "profile": config.profile,
        "model_label": "Fresh paired LeWM input-encoding screen",
        "data_hash": data_hash,
        "dataset_manifest_sha256": data_hash,
        "palette": palette.as_dict(),
        "palette_rgb": palette.as_dict()["palette_rgb"],
        "palette_sha256": palette.palette_sha256,
        "palette_source_split": palette.source_split,
        "palette_frame_index_sha256": palette.frame_index_sha256,
        "palette_selected_pixels_sha256": palette.selected_pixels_sha256,
        "palette_sampling_seed": palette.sampling_seed,
        "palette_frames_per_episode": palette.frames_per_episode,
        "palette_frame_count": palette.frame_count,
        "history_size": config.history_size,
        "batch_size": BATCH_SIZE,
        "total_steps": STEPS,
        "initialization_seed": INIT_SEED,
        "sampling_seed": SAMPLING_SEED,
        "stochastic_seed": STOCHASTIC_SEED,
        "initial_state_sha256": initial_state_sha256,
        "parameter_count": parameter_count,
        "optimizer": {
            "name": "AdamW",
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "clip": GRADIENT_CLIP,
            "schedule": "constant for declared experiment",
        },
        "world_model_updates": STEPS,
        "precision": "float32",
        "quality": "paired engineering diagnostic",
        "inference_only": False,
    }


def _checkpoint_payload(
    *,
    model: LeWorldModel,
    optimizer: torch.optim.Optimizer,
    config_payload: dict[str, object],
    metadata: Mapping[str, object],
    step: int,
    sampling_rng: torch.Generator,
    post_rng: _RNGState,
    sample_trace_hash: str,
    stochastic_trace_hash: str,
) -> dict[str, object]:
    return {
        "config": config_payload,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "torch_rng": post_rng.cpu,
        "cuda_rng": list(post_rng.cuda) if post_rng.cuda is not None else None,
        "sampling_rng": sampling_rng.get_state(),
        "stochastic_rng": {
            "cpu": post_rng.cpu,
            "cuda": list(post_rng.cuda) if post_rng.cuda is not None else None,
        },
        "batch_size": BATCH_SIZE,
        "seed": INIT_SEED,
        "initialization_seed": INIT_SEED,
        "sampling_seed": SAMPLING_SEED,
        "stochastic_seed": STOCHASTIC_SEED,
        "data_hash": metadata["data_hash"],
        "experiment": EXPERIMENT,
        "model_label": metadata["model_label"],
        "input_arm": metadata["input_arm"],
        "input_encoding": metadata["input_encoding"],
        "palette": metadata["palette"],
        "palette_rgb": metadata["palette_rgb"],
        "palette_sha256": metadata["palette_sha256"],
        "palette_source_split": metadata["palette_source_split"],
        "initial_state_sha256": metadata["initial_state_sha256"],
        "parameter_count": metadata["parameter_count"],
        "palette_frame_index_sha256": metadata["palette_frame_index_sha256"],
        "palette_selected_pixels_sha256": metadata[
            "palette_selected_pixels_sha256"
        ],
        "sample_trace_sha256": sample_trace_hash,
        "stochastic_trace_sha256": stochastic_trace_hash,
        "sample_trace_steps": step,
        "world_model_updates": step,
        "inference_only": False,
    }


def run_pair(
    dataset_root: Path,
    history_root: Path,
    run_id: str = "lewm-input-encoding-20260915-v1",
    *,
    steps: int = STEPS,
    batch_size: int = BATCH_SIZE,
    device: str = "cuda",
    threads: int = 2,
) -> dict[str, object]:
    """Train fresh matched RGB and palette world models on one T4.

    The returned ``checkpoints`` mapping points to the final raw checkpoints;
    ``pair`` points to the shared manifest.  No validation pixels are read
    during training, and no decoder or projector calibration is performed.
    """

    _validate_run_protocol(
        steps=steps,
        batch_size=batch_size,
        device=device,
        threads=threads,
    )
    torch.set_num_threads(threads)
    palette = discover_palette(dataset_root)
    if len(palette.palette_rgb) != 3:
        raise ValueError("paired input screen requires exactly three palette colors")
    dataset = LargePixelSequenceDataset(
        Path(dataset_root),
        split="train",
        history_size=LARGE_DEFAULT_HISTORY_SIZE,
        cache_size=LARGE_DEFAULT_CACHE_SIZE,
    )
    data_hash = file_hash(Path(dataset_root) / "manifest.json")
    models = _build_models(palette, device=device)
    initial_state_sha256 = {
        arm: _model_state_sha256(models[arm]) for arm in ARMS
    }
    parameter_count = {arm: _parameter_count(models[arm]) for arm in ARMS}
    if len(set(initial_state_sha256.values())) != 1:
        raise RuntimeError("paired arms do not share an initial state digest")
    if len(set(parameter_count.values())) != 1:
        raise RuntimeError("paired arms do not share a parameter count")
    optimizers = {
        arm: torch.optim.AdamW(
            models[arm].parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        for arm in ARMS
    }
    torch.manual_seed(STOCHASTIC_SEED)
    torch.cuda.manual_seed_all(STOCHASTIC_SEED)
    sampling_rng = torch.Generator(device="cpu").manual_seed(SAMPLING_SEED)
    configs = {arm: _canonical_config(models[arm].config) for arm in ARMS}
    arm_ids = {arm: f"{run_id}-{arm}" for arm in ARMS}
    metadata = {
        arm: _arm_metadata(
            run_id=arm_ids[arm],
            arm=arm,
            config=models[arm].config,
            config_payload=configs[arm],
            palette=palette,
            data_hash=data_hash,
            pair_run_id=run_id,
            initial_state_sha256=initial_state_sha256[arm],
            parameter_count=parameter_count[arm],
        )
        for arm in ARMS
    }
    runs: dict[str, Path] = {}
    for arm in ARMS:
        runs[arm] = create_run(history_root, arm_ids[arm], metadata[arm])
        atomic_json(runs[arm] / "config.json", configs[arm])
        write_status(
            runs[arm],
            state="running",
            phase="paired world-model training",
            step=0,
            total_steps=STEPS,
            message="Fresh RGB/palette pair initialized",
        )
    trace: list[dict[str, object]] = []
    stochastic_trace: list[dict[str, object]] = []
    began = time.monotonic()
    stochastic_state = _capture_rng(device)
    try:
        for step in range(1, STEPS + 1):
            batch = sample_shared_batch(dataset, BATCH_SIZE, sampling_rng)
            trace.append(batch.trace_entry(step))
            pre_hash = _rng_digest(stochastic_state)
            metrics, post_states = _paired_update(
                models,
                optimizers,
                batch,
                stochastic_state=stochastic_state,
                device=device,
            )
            post_hash = _rng_digest(post_states["rgb"])
            stochastic_trace.append(
                {
                    "step": step,
                    "pre_sha256": pre_hash,
                    "post_sha256": post_hash,
                    "arms_equal": _same_rng(
                        post_states["rgb"], post_states["palette"]
                    ),
                }
            )
            stochastic_state = post_states["rgb"]
            sample_hash = _trace_hash(trace)
            stochastic_hash = _trace_hash(stochastic_trace)
            elapsed = time.monotonic() - began
            for arm in ARMS:
                metric = {
                    "step": step,
                    **metrics[arm],
                    "sample_trace_sha256": sample_hash,
                    "stochastic_trace_sha256": stochastic_hash,
                    "updates_per_second": step / max(elapsed, 1e-9),
                    "elapsed_seconds": elapsed,
                    "phase": "paired world-model training",
                    "input_arm": arm,
                }
                append_metric(runs[arm] / "metrics.jsonl", metric)
                write_status(
                    runs[arm],
                    state="running",
                    phase="paired world-model training",
                    step=step,
                    total_steps=STEPS,
                    message="Matched native batch and stochastic state consumed",
                )
            if step in CHECKPOINT_STEPS:
                for arm in ARMS:
                    payload = _checkpoint_payload(
                        model=models[arm],
                        optimizer=optimizers[arm],
                        config_payload=configs[arm],
                        metadata=metadata[arm],
                        step=step,
                        sampling_rng=sampling_rng,
                        post_rng=post_states[arm],
                        sample_trace_hash=sample_hash,
                        stochastic_trace_hash=stochastic_hash,
                    )
                    save_checkpoint(runs[arm] / f"checkpoint-{step}.pt", payload)
                    if step == STEPS:
                        save_checkpoint(runs[arm] / "checkpoint.pt", payload)
            if step % 32 == 0 or step == STEPS:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "rgb_loss": metrics["rgb"]["loss"],
                            "palette_loss": metrics["palette"]["loss"],
                            "sample_trace_sha256": sample_hash,
                        }
                    ),
                    flush=True,
                )
        trace_hash = _trace_hash(trace)
        stochastic_hash = _trace_hash(stochastic_trace)
        for arm in ARMS:
            atomic_json(runs[arm] / "sample-trace.json", trace)
            atomic_json(runs[arm] / "stochastic-trace.json", stochastic_trace)
            atomic_json(
                runs[arm] / "report.json",
                {
                    **metadata[arm],
                    "quality_gate": "paired engineering diagnostic",
                    "steps": STEPS,
                    "global_step": STEPS,
                    "checkpoint_sha256": file_hash(runs[arm] / "checkpoint.pt"),
                    "sample_trace_sha256": trace_hash,
                    "stochastic_trace_sha256": stochastic_hash,
                    "elapsed_seconds": time.monotonic() - began,
                    "limitations": [
                        "bounded world-model input-encoding screen",
                        "no validation tuning or controller claim",
                        "raw checkpoints retained; frozen probes run separately",
                    ],
                },
            )
            write_status(
                runs[arm],
                state="completed",
                phase="paired world-model training",
                step=STEPS,
                total_steps=STEPS,
                message="Raw checkpoint ready for frozen input probes",
            )
        pair_manifest = {
            "variant": "pixel-repr-ddqn",
            "experiment": EXPERIMENT,
            "run_id": run_id,
            "arms": {arm: str(runs[arm]) for arm in ARMS},
            "checkpoints": {
                arm: str(runs[arm] / "checkpoint.pt") for arm in ARMS
            },
            "checkpoint_sha256": {
                arm: file_hash(runs[arm] / "checkpoint.pt") for arm in ARMS
            },
            "data_hash": data_hash,
            "dataset_manifest_sha256": data_hash,
            "palette": palette.as_dict(),
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "history_size": LARGE_DEFAULT_HISTORY_SIZE,
            "initialization_seed": INIT_SEED,
            "sampling_seed": SAMPLING_SEED,
            "stochastic_seed": STOCHASTIC_SEED,
            "sample_trace_sha256": trace_hash,
            "stochastic_trace_sha256": stochastic_hash,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "initial_state_sha256": initial_state_sha256,
            "parameter_count": parameter_count,
            "matched_initial_state": len(set(initial_state_sha256.values())) == 1,
            "world_model_fits": True,
            "validation_used_for_training": False,
            "decoder_feedback": False,
        }
        pair_path = Path(history_root) / f"{run_id}-pair.json"
        atomic_json(pair_path, pair_manifest)
        return {
            "checkpoints": {
                arm: runs[arm] / "checkpoint.pt" for arm in ARMS
            },
            "pair": pair_path,
        }
    except BaseException as error:
        for run in runs.values():
            write_status(
                run,
                state="failed",
                phase="paired world-model training",
                step=0,
                total_steps=STEPS,
                message=str(error),
            )
        raise


def _validate_continue_protocol(
    *,
    extra_steps: int,
    batch_size: int,
    device: str,
    threads: int,
) -> None:
    if extra_steps < 1:
        raise ValueError("continuation requires at least one extra update")
    if batch_size != BATCH_SIZE:
        raise ValueError(f"{EXPERIMENT} continuation requires batch{BATCH_SIZE}")
    if device != "cuda":
        raise ValueError(f"{EXPERIMENT} continuation requires CUDA T4")
    if threads < 1:
        raise ValueError("threads must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required: run scaled continuation on a T4")
    if "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("scaled continuation requires an NVIDIA T4")


def _single_update(
    model: LeWorldModel,
    optimizer: torch.optim.Optimizer,
    batch: SharedBatch,
    *,
    stochastic_state: _RNGState,
    device: str,
) -> tuple[dict[str, float], _RNGState]:
    """Run one palette-arm update from a pre-step stochastic state."""

    pixels = batch.pixels.to(device)
    actions = batch.actions.to(device)
    _restore_rng(stochastic_state)
    model.train()
    metrics = _train_arm_step(model, optimizer, pixels, actions)
    return metrics, _capture_rng(device)


def continue_run(
    checkpoint_path: Path,
    dataset_root: Path,
    history_root: Path,
    run_id: str,
    *,
    extra_steps: int,
    checkpoint_every: int = 1024,
    batch_size: int = BATCH_SIZE,
    device: str = "cuda",
    threads: int = 2,
) -> dict[str, object]:
    """Continue palette-arm training from a §Y-style checkpoint (AD1/AD3).

    Restores model, optimizer, sampling generator, stochastic RNG and both
    traces, then runs ``extra_steps`` more updates with the identical step
    body.  Same dataset (manifest hash must match), same hyperparameters,
    no validation reads, no decoder feedback.  Returns the new checkpoint
    path and continuation manifest path.
    """

    _validate_continue_protocol(
        extra_steps=extra_steps,
        batch_size=batch_size,
        device=device,
        threads=threads,
    )
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    torch.set_num_threads(threads)
    checkpoint_path = Path(checkpoint_path)
    base_payload: dict[str, Any] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if base_payload.get("experiment") != EXPERIMENT:
        raise ValueError("continuation requires a lewm-input-encoding-v1 payload")
    if base_payload.get("input_arm") != "palette":
        raise ValueError("scaled continuation resumes the palette arm only")
    dataset = LargePixelSequenceDataset(
        Path(dataset_root),
        split="train",
        history_size=LARGE_DEFAULT_HISTORY_SIZE,
        cache_size=LARGE_DEFAULT_CACHE_SIZE,
    )
    data_hash = file_hash(Path(dataset_root) / "manifest.json")
    if data_hash != base_payload.get("data_hash"):
        raise ValueError("continuation dataset manifest differs from base run")
    base_dir = checkpoint_path.parent
    trace = json.loads((base_dir / "sample-trace.json").read_text())
    stochastic_trace = json.loads(
        (base_dir / "stochastic-trace.json").read_text()
    )
    start_step = int(base_payload["step"])
    if len(trace) < start_step or len(stochastic_trace) < start_step:
        raise ValueError("base traces do not cover the base checkpoint step")
    trace = trace[:start_step]
    stochastic_trace = stochastic_trace[:start_step]
    if _trace_hash(trace) != base_payload.get(
        "sample_trace_sha256"
    ) or _trace_hash(stochastic_trace) != base_payload.get(
        "stochastic_trace_sha256"
    ):
        raise ValueError("base trace hashes differ from base checkpoint payload")
    base_checkpoint_sha256 = file_hash(checkpoint_path)
    model = LeWorldModel(LeWMConfig(**base_payload["config"]))
    model.load_state_dict(base_payload["model"], strict=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    optimizer.load_state_dict(base_payload["optimizer"])
    sampling_rng = torch.Generator(device="cpu")
    sampling_rng.set_state(base_payload["sampling_rng"])
    base_cuda = base_payload.get("cuda_rng")
    _restore_rng(
        _RNGState(
            cpu=base_payload["torch_rng"],
            cuda=tuple(base_cuda) if base_cuda is not None else None,
        )
    )
    metadata = {
        key: base_payload[key]
        for key in (
            "model_label",
            "input_arm",
            "input_encoding",
            "palette",
            "palette_rgb",
            "palette_sha256",
            "palette_source_split",
            "initial_state_sha256",
            "parameter_count",
            "palette_frame_index_sha256",
            "palette_selected_pixels_sha256",
        )
    }
    metadata = {
        **metadata,
        "variant": "pixel-repr-ddqn",
        "experiment": EXPERIMENT,
        "run_id": run_id,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "base_step": start_step,
        "data_hash": data_hash,
        "dataset_manifest_sha256": data_hash,
        "optimizer": {
            "name": "AdamW",
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "clip": GRADIENT_CLIP,
            "schedule": "constant for declared experiment",
        },
        "precision": "float32",
        "inference_only": False,
    }
    run = create_run(history_root, run_id, metadata)
    total_steps = start_step + extra_steps
    write_status(
        run,
        state="running",
        phase="scaled continuation training",
        step=start_step,
        total_steps=total_steps,
        message="Palette continuation restored from base checkpoint",
    )
    began = time.monotonic()
    stochastic_state = _capture_rng(device)
    try:
        for step in range(start_step + 1, total_steps + 1):
            batch = sample_shared_batch(dataset, BATCH_SIZE, sampling_rng)
            trace.append(batch.trace_entry(step))
            pre_hash = _rng_digest(stochastic_state)
            metrics, post_state = _single_update(
                model,
                optimizer,
                batch,
                stochastic_state=stochastic_state,
                device=device,
            )
            post_hash = _rng_digest(post_state)
            stochastic_trace.append(
                {
                    "step": step,
                    "pre_sha256": pre_hash,
                    "post_sha256": post_hash,
                    "arms_equal": True,
                }
            )
            stochastic_state = post_state
            sample_hash = _trace_hash(trace)
            stochastic_hash = _trace_hash(stochastic_trace)
            elapsed = time.monotonic() - began
            append_metric(
                run / "metrics.jsonl",
                {
                    "step": step,
                    **metrics,
                    "sample_trace_sha256": sample_hash,
                    "stochastic_trace_sha256": stochastic_hash,
                    "updates_per_second": (step - start_step)
                    / max(elapsed, 1e-9),
                    "elapsed_seconds": elapsed,
                    "phase": "scaled continuation training",
                    "input_arm": "palette",
                },
            )
            write_status(
                run,
                state="running",
                phase="scaled continuation training",
                step=step,
                total_steps=total_steps,
                message="Continuation batch and stochastic state consumed",
            )
            if step % checkpoint_every == 0 or step == total_steps:
                payload = _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    config_payload=base_payload["config"],
                    metadata=metadata,
                    step=step,
                    sampling_rng=sampling_rng,
                    post_rng=post_state,
                    sample_trace_hash=sample_hash,
                    stochastic_trace_hash=stochastic_hash,
                )
                # Publish running traces alongside every intermediate
                # checkpoint so an orchestrator can mirror a resumable
                # (checkpoint, traces) triple mid-chunk (B77).
                atomic_json(run / "sample-trace.json", trace)
                atomic_json(run / "stochastic-trace.json", stochastic_trace)
                save_checkpoint(run / f"checkpoint-{step}.pt", payload)
                if step == total_steps:
                    save_checkpoint(run / "checkpoint.pt", payload)
            if (step - start_step) % 32 == 0 or step == total_steps:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "palette_loss": metrics["loss"],
                            "sample_trace_sha256": sample_hash,
                        }
                    ),
                    flush=True,
                )
        trace_hash = _trace_hash(trace)
        stochastic_hash = _trace_hash(stochastic_trace)
        atomic_json(run / "sample-trace.json", trace)
        atomic_json(run / "stochastic-trace.json", stochastic_trace)
        atomic_json(
            run / "report.json",
            {
                **metadata,
                "quality_gate": "scaled continuation diagnostic",
                "steps": total_steps,
                "global_step": total_steps,
                "extra_steps": extra_steps,
                "checkpoint_sha256": file_hash(run / "checkpoint.pt"),
                "sample_trace_sha256": trace_hash,
                "stochastic_trace_sha256": stochastic_hash,
                "elapsed_seconds": time.monotonic() - began,
                "limitations": [
                    "scaled palette continuation, same frozen objective",
                    "no validation tuning or controller claim",
                    "raw checkpoint retained; frozen probes run separately",
                ],
            },
        )
        write_status(
            run,
            state="completed",
            phase="scaled continuation training",
            step=total_steps,
            total_steps=total_steps,
            message="Continuation checkpoint ready for frozen probes",
        )
        manifest = {
            **metadata,
            "steps": total_steps,
            "extra_steps": extra_steps,
            "checkpoint": str(run / "checkpoint.pt"),
            "checkpoint_sha256": file_hash(run / "checkpoint.pt"),
            "sample_trace_sha256": trace_hash,
            "stochastic_trace_sha256": stochastic_hash,
            "world_model_fits": True,
            "validation_used_for_training": False,
            "decoder_feedback": False,
        }
        manifest_path = Path(history_root) / f"{run_id}-continuation.json"
        atomic_json(manifest_path, manifest)
        return {"checkpoint": run / "checkpoint.pt", "manifest": manifest_path}
    except BaseException as error:
        write_status(
            run,
            state="failed",
            phase="scaled continuation training",
            step=start_step,
            total_steps=total_steps,
            message=str(error),
        )
        raise
