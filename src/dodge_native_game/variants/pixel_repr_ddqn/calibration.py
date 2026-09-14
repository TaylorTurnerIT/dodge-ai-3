"""Train-only encoder normalization export, separate from optimizer checkpoints."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from .dataset import PixelSequenceDataset
from .normalization_audit import capture_input
from .pretrain import load_model, save_checkpoint
from .run_artifacts import atomic_json, create_run, file_hash, write_status


def calibrate_encoder(model, dataset, device, *, batch_size=8):
    """Estimate moments with overlapping training windows weighted equally."""
    if getattr(dataset, "split", None) != "train":
        raise ValueError("calibration requires the train split")
    if not len(dataset) or batch_size < 1:
        raise ValueError("calibration needs training windows and a positive batch")
    layers = [
        (name, m)
        for name, m in model.projector.named_modules()
        if isinstance(m, nn.BatchNorm1d)
    ]
    if len(layers) != 1:
        raise ValueError("expected one encoder BatchNorm")
    name, bn = layers[0]
    model.eval().requires_grad_(False)
    values = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            pixels = torch.stack(
                [
                    dataset[i]["pixels"]
                    for i in range(start, min(start + batch_size, len(dataset)))
                ]
            ).to(device)
            values.append(
                capture_input(bn, lambda pixels=pixels: model.encode(pixels)).cpu()
            )
        # Float64 accumulation avoids cancellation in low-variance features.
        population = torch.cat(values).double()
        mean = population.mean(0)
        variance = population.var(0, unbiased=False)
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
            raise RuntimeError("nonfinite calibration moments")
        bn.running_mean.copy_(mean.to(bn.running_mean))
        bn.running_var.copy_(variance.to(bn.running_var))
    return {
        "method": "encoder-only-train-window-population-v1",
        "windows": len(dataset),
        "frame_occurrences": population.shape[0],
        "variance_estimator": "population (correction=0)",
        "changed_keys": [
            f"projector.{name}.running_mean",
            f"projector.{name}.running_var",
        ],
        "validation_used": False,
        "optimizer_updates": 0,
    }


def export_calibrated_run(
    source_run: Path,
    dataset_root: Path,
    history_root: Path,
    run_id: str,
    *,
    device="cuda",
) -> Path:
    checkpoint = source_run / "checkpoint.pt"
    parent_hash = file_hash(checkpoint)
    model, payload = load_model(checkpoint)
    if payload.get("inference_only"):
        raise ValueError("source must be an uncalibrated optimizer checkpoint")
    if payload["profile"] == "reference" and (
        device != "cuda"
        or not torch.cuda.is_available()
        or "T4" not in torch.cuda.get_device_name(0)
    ):
        raise RuntimeError("reference calibration requires Colab T4")
    data_hash = file_hash(dataset_root / "manifest.json")
    if data_hash != payload["data_hash"]:
        raise ValueError("calibration dataset differs from training dataset")
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model = model.to(device)
    dataset = PixelSequenceDataset(
        dataset_root, split="train", history_size=model.config.history_size
    )
    metadata = calibrate_encoder(model, dataset, device)
    for key, value in model.state_dict().items():
        if key not in metadata["changed_keys"] and not torch.equal(
            before[key], value.detach().cpu()
        ):
            raise RuntimeError(f"calibration altered non-encoder-stat tensor: {key}")
    if file_hash(checkpoint) != parent_hash:
        raise RuntimeError("calibration changed source checkpoint")
    metadata.update(parent_checkpoint_sha256=parent_hash, data_hash=data_hash)
    label = "Encoder-calibrated inference copy; gameplay quality unproven"
    manifest = {
        key: payload[key]
        for key in ("profile", "experiment", "seed", "batch_size", "data_hash")
        if key in payload
    }
    manifest.update(
        variant="pixel-repr-ddqn",
        model_label=label,
        inference_only=True,
        calibration=metadata,
        device=device,
        config=asdict(model.config),
        source_model_step=payload["step"],
    )
    run = create_run(history_root, run_id, manifest)
    derived = {
        key: payload[key]
        for key in (
            "config",
            "profile",
            "experiment",
            "seed",
            "batch_size",
            "data_hash",
            "step",
        )
        if key in payload
    }
    derived.update(
        model=model.state_dict(),
        model_label=label,
        inference_only=True,
        calibration=metadata,
    )
    save_checkpoint(run / "checkpoint.pt", derived)
    metadata["checkpoint_sha256"] = file_hash(run / "checkpoint.pt")
    atomic_json(run / "calibration.json", metadata)
    atomic_json(run / "config.json", asdict(model.config))
    atomic_json(
        run / "report.json",
        {"quality_gate": "inference-only diagnostic", "calibration": metadata},
    )
    write_status(
        run,
        state="completed",
        phase="Encoder calibration ready",
        step=0,
        total_steps=0,
        message="Training-only moments; original checkpoint preserved",
    )
    return run
