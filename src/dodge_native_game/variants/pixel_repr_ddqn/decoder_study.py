"""Fit and compare frozen pixel decoders against one calibrated LeWM."""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from . import pretrain, probe
from .dataset import PixelSequenceDataset
from .pixel_diagnostics import evaluate_pixels
from .query_decoder import QueryPixelDecoder
from .run_artifacts import (
    append_metric,
    atomic_json,
    create_run,
    file_hash,
    write_status,
)

__all__ = ["CachedDecoderFit", "cache_train_windows", "fit_cached_decoder", "run_study"]

_EXPERIMENT = "practice-batch32-v1"
_WORLD_STEP = 512
_WORLD_BATCH_SIZE = 32
_DECODER_BATCH_SIZE = 8
_DECODER_STEPS = 256
_EXTENDED_DECODER_STEPS = 512
_FINAL_DECODER_STEPS = 2048
_ULTIMATE_DECODER_STEPS = 8192
_LATENT_DIM = 192
_OUTPUT_SIZE = 128
_SAMPLING_SEED = 903
_DECODER_SEED = 904

_RESUME_STEPS = {
    _EXTENDED_DECODER_STEPS: _DECODER_STEPS,
    _FINAL_DECODER_STEPS: _EXTENDED_DECODER_STEPS,
    _ULTIMATE_DECODER_STEPS: _FINAL_DECODER_STEPS,
}


@dataclass(frozen=True)
class CachedDecoderFit:
    """Metrics, retained weights, and continuation state from one fit."""

    metrics: list[dict[str, float | int | str]]
    snapshots: dict[int, dict[str, torch.Tensor]]
    optimizer_state: dict[str, Any]
    generator_state: torch.Tensor
    torch_rng_state: torch.Tensor
    cuda_rng_state: list[torch.Tensor] | None


def _rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]


def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _clone_checkpoint_value(value: Any) -> Any:
    """Copy checkpoint state to CPU without retaining live module storage."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_checkpoint_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_checkpoint_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_checkpoint_value(item) for item in value)
    return value


def cache_train_windows(
    model: nn.Module,
    train_dataset: PixelSequenceDataset,
    *,
    device: torch.device | str,
    batch_size: int = _DECODER_BATCH_SIZE,
    output_size: int = _OUTPUT_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode every train window once and return detached CPU caches.

    The target cache intentionally keeps native 128x128 RGB values.  The
    decoder study never feeds a pre-CLS tensor and never re-encodes train
    windows while fitting either decoder.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if output_size != _OUTPUT_SIZE:
        raise ValueError("decoder study requires native 128x128 targets")
    if getattr(train_dataset, "split", None) != "train":
        raise ValueError("decoder cache must use the train split")
    count = len(train_dataset)
    if count < 1:
        raise ValueError("train_dataset must contain at least one window")

    target_device = torch.device(device)
    model.eval().requires_grad_(False)
    latents: torch.Tensor | None = None
    targets: torch.Tensor | None = None
    with torch.no_grad():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            samples = [train_dataset[index] for index in range(start, stop)]
            pixels = torch.stack(
                [torch.as_tensor(sample["pixels"]) for sample in samples]
            )
            if pixels.dtype != torch.uint8 or pixels.ndim != 5:
                raise ValueError("train pixels must be uint8 windows")
            if pixels.shape[2:] != (3, 128, 128):
                raise ValueError("train pixels must have native shape (3,128,128)")
            encoded = model.encode(pixels.to(target_device))
            if encoded.ndim != 3 or encoded.shape[:2] != pixels.shape[:2]:
                raise ValueError(
                    "model.encode must return shape (B,T,D), "
                    f"got {tuple(encoded.shape)}"
                )
            encoded = encoded.detach().cpu().float()
            rgb = pixels.float().div(255.0)
            if latents is None:
                latents = torch.empty(
                    (count, encoded.shape[1], encoded.shape[2]), dtype=torch.float32
                )
                targets = torch.empty(
                    (count, rgb.shape[1], 3, output_size, output_size),
                    dtype=torch.float32,
                )
            if encoded.shape[1:] != latents.shape[1:]:
                raise ValueError("model.encode returned inconsistent latent shapes")
            assert targets is not None
            latents[start:stop].copy_(encoded)
            targets[start:stop].copy_(rgb)
    assert latents is not None and targets is not None
    return latents, targets


def _decoder_output(
    prediction: torch.Tensor,
    *,
    batch: int,
    output_size: int,
) -> torch.Tensor:
    expected = 3 * output_size * output_size
    if prediction.ndim == 2:
        if prediction.shape != (batch, expected):
            raise ValueError(
                f"decoder output must have shape {(batch, expected)}, "
                f"got {tuple(prediction.shape)}"
            )
        return prediction.reshape(batch, 3, output_size, output_size)
    if prediction.ndim == 4 and prediction.shape == (
        batch,
        3,
        output_size,
        output_size,
    ):
        return prediction
    raise ValueError(
        "decoder output must be flat RGB or (B,3,128,128), "
        f"got {tuple(prediction.shape)}"
    )


def fit_cached_decoder(
    decoder: nn.Module,
    latents: torch.Tensor,
    targets: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    seed: int = _DECODER_SEED,
    sampling_seed: int = _SAMPLING_SEED,
    discard_indices: int = 2,
    snapshot_steps: tuple[int, ...] = (),
    resume_state: dict[str, Any] | None = None,
    on_step: Callable[[dict[str, float | int | str]], None] | None = None,
) -> CachedDecoderFit:
    """Fit one decoder from detached caches without touching the world model.

    ``steps`` is the total optimizer step after a possible continuation.  A
    resume state therefore starts logging at ``resume_state["steps"] + 1``
    and continues the same AdamW and CPU sampling-generator streams.
    """

    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    if discard_indices < 0:
        raise ValueError("discard_indices must be non-negative")
    if latents.ndim != 3:
        raise ValueError("latents must have shape (windows,time,dimension)")
    if targets.ndim != 5 or targets.shape[:2] != latents.shape[:2]:
        raise ValueError("targets must have shape (windows,time,3,height,width)")
    if targets.shape[2:] != (3, _OUTPUT_SIZE, _OUTPUT_SIZE):
        raise ValueError("targets must be native 128x128 RGB frames")
    if len(latents) < batch_size:
        raise ValueError("cache does not contain a complete decoder batch")
    if any(step < 1 or step > steps for step in snapshot_steps):
        raise ValueError("snapshot steps must be within the fit")

    try:
        decoder_device = next(decoder.parameters()).device
    except StopIteration as error:
        raise ValueError("decoder must have trainable parameters") from error
    decoder.train()
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=1e-3, weight_decay=0.01
    )
    start_step = 0
    if resume_state is None:
        generator = torch.Generator(device="cpu").manual_seed(sampling_seed)
        if discard_indices:
            torch.randint(len(latents), (discard_indices,), generator=generator)
    else:
        try:
            start_step = int(resume_state["steps"])
            optimizer_state = resume_state["optimizer"]
            generator_state = resume_state.get("generator_state")
            if generator_state is None:
                generator_state = resume_state.get("sampling_rng")
            torch_rng_state = resume_state["torch_rng"]
        except KeyError as error:
            raise ValueError(
                "decoder resume state must contain steps, optimizer, "
                "generator_state, and torch_rng"
            ) from error
        if start_step < 1 or start_step >= steps:
            raise ValueError("decoder resume must start before the requested total")
        if not isinstance(generator_state, torch.Tensor):
            raise ValueError("decoder generator_state must be a tensor")
        if not isinstance(torch_rng_state, torch.Tensor):
            raise ValueError("decoder torch_rng must be a tensor")
        optimizer.load_state_dict(optimizer_state)
        generator = torch.Generator(device="cpu")
        generator.set_state(generator_state.detach().cpu())

    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    metrics: list[dict[str, float | int | str]] = []
    began = time.monotonic()
    with torch.random.fork_rng(devices=_rng_devices(decoder_device)):
        if resume_state is None:
            torch.manual_seed(seed)
        else:
            torch.set_rng_state(torch_rng_state.detach().cpu())
            if decoder_device.type == "cuda" and resume_state.get("cuda_rng"):
                torch.cuda.set_rng_state_all(resume_state["cuda_rng"])
        for step in range(start_step + 1, steps + 1):
            indices = torch.randint(
                len(latents), (batch_size,), generator=generator
            )
            latent_batch = latents.index_select(0, indices).reshape(
                -1, latents.shape[-1]
            ).detach().to(decoder_device)
            target_batch = targets.index_select(0, indices).reshape(
                -1, 3, _OUTPUT_SIZE, _OUTPUT_SIZE
            ).detach().to(decoder_device)
            prediction = _decoder_output(
                decoder(latent_batch),
                batch=int(latent_batch.shape[0]),
                output_size=_OUTPUT_SIZE,
            )
            loss = torch.nn.functional.mse_loss(prediction, target_batch)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite decoder loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            elapsed = time.monotonic() - began
            metric: dict[str, float | int | str] = {
                "step": step,
                "decoder_loss": float(loss.detach()),
                "loss": float(loss.detach()),
                "updates_per_second": (step - start_step) / max(elapsed, 1e-9),
                "phase": "diagnostic decoder fitting",
            }
            metrics.append(metric)
            if step in snapshot_steps:
                snapshots[step] = _clone_state(decoder)
            if on_step is not None:
                on_step(metric)
        final_torch_rng_state = torch.get_rng_state().detach().cpu().clone()
        final_cuda_rng_state = (
            [state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()]
            if decoder_device.type == "cuda"
            else None
        )
    decoder.eval()
    return CachedDecoderFit(
        metrics=metrics,
        snapshots=snapshots,
        optimizer_state=_clone_checkpoint_value(optimizer.state_dict()),
        generator_state=generator.get_state().detach().cpu().clone(),
        torch_rng_state=final_torch_rng_state,
        cuda_rng_state=final_cuda_rng_state,
    )


def _fresh_query_decoder(
    dimension: int, device: torch.device | str
) -> nn.Module:
    target_device = torch.device(device)
    with torch.random.fork_rng(devices=_rng_devices(target_device)):
        torch.manual_seed(_DECODER_SEED)
        decoder = QueryPixelDecoder(dimension)
    return decoder.to(target_device)


def _resolve_checkpoint(path: Path) -> Path:
    candidate = path / "checkpoint.pt" if path.is_dir() else path
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _resolve_decoder_checkpoint(path: Path) -> Path:
    candidate = path / "decoder.pt" if path.is_dir() else path
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _validate_resume_step(
    payload: dict[str, Any], *, total_steps: int
) -> None:
    required_steps = _RESUME_STEPS.get(total_steps)
    if required_steps is None:
        raise ValueError(f"{total_steps}-update decoder study cannot resume")
    if payload.get("steps") != required_steps:
        raise ValueError(
            f"decoder resume checkpoint must contain {required_steps} updates"
        )


def _link_checkpoint(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _require_protocol(
    payload: dict[str, Any],
    *,
    device: str,
    model: nn.Module,
) -> str:
    if payload.get("inference_only") is not True:
        raise ValueError("decoder study requires an inference-only checkpoint")
    if payload.get("experiment") != _EXPERIMENT:
        raise ValueError("decoder study requires practice-batch32-v1")
    if payload.get("step") != _WORLD_STEP:
        raise ValueError("decoder study requires world-model step 512")
    if payload.get("batch_size") != _WORLD_BATCH_SIZE:
        raise ValueError("decoder study requires the batch32 world-model checkpoint")
    if payload.get("profile") != "reference":
        raise ValueError("decoder study requires the reference model profile")
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("reference decoder study requires CUDA")
    gpu = torch.cuda.get_device_name(0)
    if "T4" not in gpu:
        raise RuntimeError(f"reference decoder study requires an actual T4, got {gpu}")
    config = getattr(model, "config", None)
    if getattr(config, "embed_dim", None) != _LATENT_DIM:
        raise ValueError("decoder study requires latent dimension 192")
    return gpu


def _assert_unchanged(
    checkpoint: Path,
    checkpoint_hash: str,
    manifest: Path,
    data_hash: str,
    model: nn.Module,
    state: dict[str, torch.Tensor],
) -> None:
    if file_hash(checkpoint) != checkpoint_hash:
        raise RuntimeError("world-model checkpoint changed during decoder study")
    if file_hash(manifest) != data_hash:
        raise RuntimeError("dataset manifest changed during decoder study")
    for key, value in model.state_dict().items():
        if key not in state or not torch.equal(state[key], value.detach().cpu()):
            raise RuntimeError(f"frozen world model changed: {key}")


def _write_fit_artifacts(
    run: Path,
    *,
    fit: CachedDecoderFit,
    snapshot_step: int,
    decoder_kind: str,
    latent_dim: int,
    world_hash: str,
    data_hash: str,
    model: nn.Module,
    train: PixelSequenceDataset,
    validation: PixelSequenceDataset,
    payload: dict[str, Any],
    checkpoint_hash: str,
    device: str,
) -> None:
    if decoder_kind != "query":
        raise ValueError("decoder study only supports the query decoder")
    state = fit.snapshots[snapshot_step]
    decoder = _fresh_query_decoder(latent_dim, device)
    decoder.load_state_dict(state, strict=True)
    decoder.eval().requires_grad_(False)
    pretrain.save_checkpoint(
        run / "decoder.pt",
        {
            "model": state,
            "optimizer": fit.optimizer_state,
            "generator_state": fit.generator_state,
            # Keep the project training-state spelling alongside the explicit
            # field above for artifact readers that use the world-model key.
            "sampling_rng": fit.generator_state,
            "torch_rng": fit.torch_rng_state,
            "cuda_rng": fit.cuda_rng_state,
            "latent_dim": latent_dim,
            "steps": snapshot_step,
            "world_model_sha256": world_hash,
            "data_hash": data_hash,
            "decoder_kind": decoder_kind,
            "batch_size": _DECODER_BATCH_SIZE,
            "seed": _DECODER_SEED,
            "sampling_seed": _SAMPLING_SEED,
            "discard_indices": 2,
        },
    )
    decoder_hash = file_hash(run / "decoder.pt")
    for metric in fit.metrics:
        append_metric(run / "decoder_metrics.jsonl", metric)

    common = {
        "checkpoint_sha256": checkpoint_hash,
        "decoder_sha256": decoder_hash,
        "data_hash": data_hash,
        "decoder_kind": decoder_kind,
        "decoder_steps": snapshot_step,
        "diagnostic_only": True,
    }
    train_checks = evaluate_pixels(
        model, decoder, train, train, device, output_size=_OUTPUT_SIZE
    )
    train_checks.update(common, split="train")
    atomic_json(run / "train_pixel_diagnostics.json", train_checks)
    validation_checks = evaluate_pixels(
        model, decoder, train, validation, device, output_size=_OUTPUT_SIZE
    )
    validation_checks.update(common, split="validation")
    atomic_json(run / "pixel_diagnostics.json", validation_checks)
    validation_32 = evaluate_pixels(model, decoder, train, validation, device)
    validation_32.update(common, split="validation", output_size=32)
    atomic_json(run / "pixel_diagnostics_32.json", validation_32)

    visualization_payload = dict(payload)
    visualization_payload["model_label"] = (
        f"Frozen {decoder_kind} decoder ({snapshot_step} updates) — diagnostic only"
    )
    visualization_payload["step"] = payload["step"]
    snapshots = probe.export_visualizations(
        model,
        decoder,
        validation,
        visualization_payload,
        checkpoint_hash,
        output_size=_OUTPUT_SIZE,
        decoder_steps=snapshot_step,
    )
    if len(snapshots) != 8:
        raise RuntimeError("decoder study requires exactly eight validation snapshots")
    atomic_json(run / "visualizations.json", snapshots)
    atomic_json(run / "visualization.json", snapshots[0])
    atomic_json(
        run / "report.json",
        {
            "quality_gate": "inference-only diagnostic",
            "diagnostic_only": True,
            "decoder_kind": decoder_kind,
            "decoder_steps": snapshot_step,
            "world_model_step": payload["step"],
            "checkpoint_sha256": checkpoint_hash,
            "decoder_sha256": decoder_hash,
            "data_hash": data_hash,
            "limitations": [
                "decoder fit uses training windows only",
                "world-model tensors receive no updates",
                "no gameplay, control, or semantic quality claim",
            ],
        },
    )
    write_status(
        run,
        state="completed",
        phase="frozen decoder diagnostics ready",
        step=snapshot_step,
        total_steps=snapshot_step,
        message="Inference-only diagnostic artifacts ready",
    )


def run_study(
    checkpoint: Path,
    dataset_root: Path,
    history_root: Path,
    run_id: str,
    device: str = "cuda",
    steps: int = _DECODER_STEPS,
    batch_size: int = _DECODER_BATCH_SIZE,
    resume_decoder: Path | None = None,
) -> list[Path]:
    """Fit one query decoder, optionally continuing its declared prior checkpoint."""

    if steps not in {
        _DECODER_STEPS,
        _EXTENDED_DECODER_STEPS,
        _FINAL_DECODER_STEPS,
        _ULTIMATE_DECODER_STEPS,
    }:
        raise ValueError(
            "decoder study requires a total of 256, 512, 2048, or 8192 updates"
        )
    if batch_size != _DECODER_BATCH_SIZE:
        raise ValueError("decoder study requires batch8")
    if steps == _DECODER_STEPS and resume_decoder is not None:
        raise ValueError("256-update decoder study cannot resume a checkpoint")
    if steps == _EXTENDED_DECODER_STEPS and resume_decoder is None:
        raise ValueError("512-update decoder study requires a 256-update checkpoint")
    if steps == _FINAL_DECODER_STEPS and resume_decoder is None:
        raise ValueError("2048-update decoder study requires a 512-update checkpoint")
    if steps == _ULTIMATE_DECODER_STEPS and resume_decoder is None:
        raise ValueError("8192-update decoder study requires a 2048-update checkpoint")
    checkpoint_path = _resolve_checkpoint(Path(checkpoint))
    dataset_path = Path(dataset_root)
    manifest_path = dataset_path / "manifest.json"
    checkpoint_hash = file_hash(checkpoint_path)
    data_hash = file_hash(manifest_path)
    model, payload = pretrain.load_model(checkpoint_path)
    gpu = _require_protocol(payload, device=device, model=model)
    if payload.get("data_hash") != data_hash:
        raise ValueError("decoder study dataset differs from model training dataset")
    model = model.to(device).eval().requires_grad_(False)
    model_state = _clone_state(model)
    history_path = Path(history_root)
    train = PixelSequenceDataset(
        dataset_path, split="train", history_size=model.config.history_size
    )
    validation = PixelSequenceDataset(
        dataset_path, split="validation", history_size=model.config.history_size
    )
    if len(train) != 168:
        raise ValueError(f"decoder study requires 168 train windows, got {len(train)}")
    if len(validation) != 56:
        raise ValueError(
            "decoder study requires exactly 56 validation windows, "
            f"got {len(validation)}"
        )
    latents, targets = cache_train_windows(
        model, train, device=device, batch_size=batch_size
    )

    resume_path: Path | None = None
    resume_payload: dict[str, Any] | None = None
    if resume_decoder is not None:
        resume_path = _resolve_decoder_checkpoint(Path(resume_decoder))
        loaded = torch.load(resume_path, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict):
            raise ValueError("decoder resume checkpoint must contain a mapping")
        resume_payload = loaded
        if resume_payload.get("decoder_kind") != "query":
            raise ValueError("decoder resume checkpoint must be a query decoder")
        _validate_resume_step(resume_payload, total_steps=steps)
        if resume_payload.get("latent_dim") != _LATENT_DIM:
            raise ValueError("decoder resume latent dimension mismatch")
        if resume_payload.get("batch_size") != _DECODER_BATCH_SIZE:
            raise ValueError("decoder resume batch size mismatch")
        if resume_payload.get("world_model_sha256") != checkpoint_hash:
            raise ValueError("decoder resume world-model checkpoint mismatch")
        if resume_payload.get("data_hash") != data_hash:
            raise ValueError("decoder resume dataset manifest mismatch")
        for key in ("model", "optimizer", "generator_state", "torch_rng"):
            if key not in resume_payload:
                raise ValueError(f"decoder resume checkpoint missing {key}")

    decoder = _fresh_query_decoder(_LATENT_DIM, device)
    if resume_payload is not None:
        decoder.load_state_dict(resume_payload["model"], strict=True)

    label = f"Frozen query decoder ({steps} updates) — diagnostic only"
    run = create_run(
        history_path,
        f"{run_id}-query{steps}",
        {
            "variant": "pixel-repr-ddqn",
            "experiment": "decoder-study-v1",
            "profile": payload["profile"],
            "model_label": label,
            "decoder_kind": "query",
            "decoder_steps": steps,
            "step": payload["step"],
            "world_model_step": payload["step"],
            "world_model_sha256": checkpoint_hash,
            "data_hash": data_hash,
            "dataset_root": str(dataset_path.resolve()),
            "batch_size": batch_size,
            "seed": _DECODER_SEED,
            "sampling_seed": _SAMPLING_SEED,
            "device": device,
            "gpu": gpu,
            "output_size": _OUTPUT_SIZE,
            "validation_windows": len(validation),
            "quality": "inference-only diagnostic",
            "inference_only": True,
            "total_steps": steps,
            "resume_decoder_sha256": (
                file_hash(resume_path) if resume_path is not None else None
            ),
            "config": payload.get("config"),
        },
    )
    _link_checkpoint(checkpoint_path, run / "checkpoint.pt")
    atomic_json(run / "config.json", payload.get("config", {}))
    initial_step = int(resume_payload["steps"]) if resume_payload else 0
    write_status(
        run,
        state="running",
        phase="diagnostic decoder fitting (queued)",
        step=initial_step,
        total_steps=steps,
        message="World model frozen; decoder fit pending",
    )

    def progress(metric: dict[str, float | int | str]) -> None:
        step = int(metric["step"])
        dashboard_metric = {
            "step": step,
            "loss": metric["decoder_loss"],
            "updates_per_second": metric["updates_per_second"],
            "phase": metric["phase"],
        }
        append_metric(run / "metrics.jsonl", dashboard_metric)
        if step == initial_step + 1 or step % 32 == 0 or step == steps:
            write_status(
                run,
                state="running",
                phase="diagnostic decoder fitting",
                step=step,
                total_steps=steps,
                message="World model frozen; decoder-only updates",
            )

    fit = fit_cached_decoder(
        decoder,
        latents,
        targets,
        steps=steps,
        batch_size=batch_size,
        seed=_DECODER_SEED,
        sampling_seed=_SAMPLING_SEED,
        discard_indices=2,
        snapshot_steps=(steps,),
        resume_state=resume_payload,
        on_step=progress,
    )
    _write_fit_artifacts(
        run,
        fit=fit,
        snapshot_step=steps,
        decoder_kind="query",
        latent_dim=_LATENT_DIM,
        world_hash=checkpoint_hash,
        data_hash=data_hash,
        model=model,
        train=train,
        validation=validation,
        payload=payload,
        checkpoint_hash=checkpoint_hash,
        device=device,
    )

    _assert_unchanged(
        checkpoint_path,
        checkpoint_hash,
        manifest_path,
        data_hash,
        model,
        model_state,
    )
    return [run]
