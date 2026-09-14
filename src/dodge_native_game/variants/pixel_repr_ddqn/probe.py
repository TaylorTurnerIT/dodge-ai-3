"""Fit a separate pixel decoder with a frozen LeWM and export held-out views."""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from .dataset import PixelSequenceDataset
from .pretrain import batch_from_dataset, load_model, save_checkpoint
from .run_artifacts import append_metric, atomic_json, file_hash, write_status


def png(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu()
    if array.dtype != torch.uint8:
        array = (array.clamp(0, 1) * 255).round().to(torch.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array.permute(1, 2, 0).numpy()).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def fit_probe(
    run: Path,
    dataset_root: Path,
    *,
    steps: int = 32,
    batch_size: int = 4,
    device: str = "cuda",
) -> None:
    if not 1 <= steps <= 256:
        raise ValueError("decoder fitting requires 1..256 updates")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required: fit diagnostics on Colab T4")
    checkpoint = run / "checkpoint.pt"
    checkpoint_hash = file_hash(checkpoint)
    model, payload = load_model(checkpoint)
    practice = payload.get("experiment") in {
        "practice-overfit-v1",
        "practice-diverse-v1",
    }
    if steps > 32 and not practice:
        raise ValueError("decoder smoke requires 1..32 updates")
    if practice and (steps, batch_size, device) != (256, 8, "cuda"):
        raise ValueError("practice decoder requires 256 updates, batch8, CUDA")
    if file_hash(dataset_root / "manifest.json") != payload["data_hash"]:
        raise ValueError("diagnostic dataset differs from model training dataset")
    model = model.to(device).eval().requires_grad_(False)
    before = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    train = PixelSequenceDataset(
        dataset_root, split="train", history_size=model.config.history_size
    )
    validation = PixelSequenceDataset(
        dataset_root, split="validation", history_size=model.config.history_size
    )
    if not len(train) or not len(validation):
        raise ValueError("both dataset splits need valid windows")
    rng = torch.Generator().manual_seed(903)
    torch.manual_seed(904)
    example, _ = batch_from_dataset(train, 2, rng)
    with torch.no_grad():
        dimension = model.encode(example.to(device)).shape[-1]
    decoder = nn.Sequential(
        nn.Linear(dimension, 256), nn.GELU(), nn.Linear(256, 3 * 32 * 32), nn.Sigmoid()
    ).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-3)
    for step in range(1, steps + 1):
        pixels, _ = batch_from_dataset(train, batch_size, rng)
        pixels = pixels.to(device)
        with torch.no_grad():
            latent = model.encode(pixels).flatten(0, 1)
            target = F.interpolate(
                pixels.flatten(0, 1).float() / 255, size=(32, 32), mode="area"
            )
        prediction = decoder(latent).view(-1, 3, 32, 32)
        loss = F.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite decoder loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        append_metric(
            run / "decoder_metrics.jsonl",
            {"step": step, "decoder_loss": float(loss.detach())},
        )
        write_status(
            run,
            state="running",
            phase="frozen decoder smoke",
            step=step,
            total_steps=steps,
            message="Diagnostic fitting; world model frozen",
        )
    decoder.eval()
    snapshots = []
    with torch.no_grad():
        for index in np.linspace(
            0, len(validation) - 1, min(8, len(validation)), dtype=int
        ):
            sample = validation[int(index)]
            pixels = sample["pixels"].unsqueeze(0).to(device)
            actions = sample["actions"].unsqueeze(0).to(device)
            z, attention = model.encode_with_attention(pixels)
            predicted = model.predict(z[:, :-1], actions)[:, -1]
            current = z[:, -2]
            decoded = decoder(torch.cat([current, predicted], dim=0)).view(2, 3, 32, 32)
            values = current[0].cpu().tolist()
            snapshots.append(
                {
                    "step": payload["step"],
                    "model_label": payload["model_label"],
                    "diagnostic_only": True,
                    "frames": [
                        {"label": "Observed current", "image": png(pixels[0, -2])},
                        {"label": "Observed next", "image": png(pixels[0, -1])},
                        {
                            "label": "Decoded current (diagnostic)",
                            "image": png(decoded[0]),
                        },
                        {
                            "label": "Decoded prediction (diagnostic)",
                            "image": png(decoded[1]),
                        },
                    ],
                    "attention": {
                        "values": attention[0, -2].cpu().tolist(),
                        "label": "Current CLS attention; not an object mask",
                    },
                    "features": {
                        "values": [values],
                        "label": "Latent coordinates; not spatial positions",
                    },
                    "latent": {
                        "actual": z[0, -1].cpu().tolist(),
                        "predicted": predicted[0].cpu().tolist(),
                    },
                    "metadata": {
                        "checkpoint_sha256": checkpoint_hash,
                        "validation_window": int(index),
                        "action": int(actions[0, -1]),
                        "latent_mse": float(F.mse_loss(predicted, z[:, -1])),
                        "decoder_steps": steps,
                    },
                }
            )
    for key, value in model.state_dict().items():
        if not torch.equal(before[key], value.detach().cpu()):
            raise RuntimeError(f"frozen world model changed: {key}")
    if file_hash(checkpoint) != checkpoint_hash:
        raise RuntimeError("world-model checkpoint changed during diagnostics")
    save_checkpoint(
        run / "decoder.pt",
        {
            "model": decoder.state_dict(),
            "latent_dim": dimension,
            "steps": steps,
            "world_model_sha256": checkpoint_hash,
        },
    )
    atomic_json(run / "visualizations.json", snapshots)
    atomic_json(run / "visualization.json", snapshots[0])
    write_status(
        run,
        state="completed",
        phase="Practice diagnostics ready" if practice else "MVP ready",
        step=steps,
        total_steps=steps,
        message="Held-out diagnostics ready; no gameplay quality claim",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    fit_probe(
        args.run,
        args.dataset,
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
