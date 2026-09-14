"""Bounded LeWM training for the independently observable MVP.

The two-term loss is implemented in model.py against the pinned LeWM source.
This runner adds dataset, checkpoint and read-only dashboard integration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .dataset import PixelSequenceDataset
from .model import LeWMConfig, LeWorldModel
from .run_artifacts import (
    append_metric,
    atomic_json,
    create_run,
    file_hash,
    write_status,
)

DEFAULT_ROOT = Path("history/dodge/gymnasium/pixel-repr-ddqn")


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_model(checkpoint: Path) -> tuple[LeWorldModel, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = LeWorldModel(LeWMConfig(**payload["config"]))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, payload


def batch_from_dataset(
    dataset: PixelSequenceDataset, size: int, rng: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randint(len(dataset), (size,), generator=rng).tolist()
    samples = [dataset[index] for index in indices]
    return (
        torch.stack([sample["pixels"] for sample in samples]),
        torch.stack([sample["actions"] for sample in samples]),
    )


def train(
    *,
    dataset_root: Path,
    history_root: Path = DEFAULT_ROOT,
    run_id: str = "lewm-mvp",
    profile: str = "reference",
    steps: int = 32,
    batch_size: int = 4,
    seed: int = 42,
    threads: int = 2,
    resume: Path | None = None,
    device: str = "cuda",
) -> Path:
    if not 1 <= steps <= 32:
        raise ValueError("MVP training is bounded to 1..32 updates per run")
    if batch_size < 2 or batch_size > 128:
        raise ValueError("batch_size must be in 2..128 for BatchNorm/SIGReg")
    if threads < 1:
        raise ValueError("threads must be positive")
    if profile not in {"tiny", "reference"}:
        raise ValueError("profile must be tiny or reference")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required: run training on Colab T4")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    config = LeWMConfig.tiny() if profile == "tiny" else LeWMConfig.reference()
    dataset = PixelSequenceDataset(
        dataset_root, split="train", history_size=config.history_size
    )
    if not len(dataset):
        raise ValueError("dataset contains no valid training windows")
    data_hash = file_hash(dataset_root / "manifest.json")
    model = LeWorldModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    sampling_rng = torch.Generator().manual_seed(seed + 1)
    start = 0
    parent_hash = None
    if resume is not None:
        previous = torch.load(resume, map_location="cpu", weights_only=True)
        if previous["config"] != asdict(config) or previous["data_hash"] != data_hash:
            raise ValueError("checkpoint configuration/dataset mismatch")
        if previous.get("batch_size") != batch_size or previous.get("seed") != seed:
            raise ValueError("checkpoint batch size/seed mismatch")
        model.load_state_dict(previous["model"], strict=True)
        optimizer.load_state_dict(previous["optimizer"])
        sampling_rng.set_state(previous["sampling_rng"])
        torch.set_rng_state(previous["torch_rng"])
        if device == "cuda" and previous.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(previous["cuda_rng"])
        start = previous["step"]
        parent_hash = file_hash(resume)
    label = (
        "Tiny engineering smoke — not validated for gameplay"
        if profile == "tiny"
        else "Reference architecture smoke — not validated for gameplay"
    )
    run = create_run(
        history_root,
        run_id,
        {
            "variant": "pixel-repr-ddqn",
            "profile": profile,
            "model_label": label,
            "config": asdict(config),
            "data_hash": data_hash,
            "dataset_root": str(dataset_root.resolve()),
            "seed": seed,
            "batch_size": batch_size,
            "device": device,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "precision": "float32",
            "torch_version": str(torch.__version__),
            "total_steps": steps,
            "parent_checkpoint_sha256": parent_hash,
            "optimizer": {
                "name": "AdamW",
                "lr": 5e-5,
                "weight_decay": 1e-3,
                "clip": 1.0,
                "schedule": "constant for bounded MVP smoke",
            },
            "upstream_commit": "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac",
            "quality": "engineering-only",
            "actions": "one-hot nine discrete actions",
        },
    )
    atomic_json(run / "config.json", asdict(config))
    began = time.monotonic()
    model.train()
    offset = 0
    try:
        for offset in range(1, steps + 1):
            pixels, actions = batch_from_dataset(dataset, batch_size, sampling_rng)
            pixels, actions = pixels.to(device), actions.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = model.compute_loss(pixels, actions)
            if not torch.isfinite(losses["loss"]):
                raise RuntimeError("nonfinite LeWM training loss")
            losses["loss"].backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad):
                raise RuntimeError("nonfinite LeWM gradient norm")
            optimizer.step()
            elapsed = time.monotonic() - began
            # Eval-only diagnostics cannot update BatchNorm or training random state.
            with torch.random.fork_rng(), torch.no_grad():
                model.eval()
                z = model.encode(pixels).flatten(0, 1)
                spread = float(z.std(dim=0, unbiased=False).mean())
                singular = torch.linalg.svdvals(z - z.mean(0))
                probability = singular / singular.sum().clamp_min(1e-12)
                rank = float(
                    torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum())
                )
            model.train()
            metric = {
                "step": start + offset,
                "loss": float(losses["loss"].detach()),
                "pred_loss": float(losses["pred_loss"].detach()),
                "sigreg_loss": float(losses["sigreg_loss"].detach()),
                "grad_norm": float(grad),
                "feature_std": spread,
                "feature_rank": rank,
                "updates_per_second": offset / max(elapsed, 1e-9),
                "elapsed_seconds": elapsed,
                "phase": "world-model smoke",
            }
            if not all(
                math.isfinite(value)
                for value in metric.values()
                if isinstance(value, float)
            ):
                raise RuntimeError("nonfinite diagnostic")
            append_metric(run / "metrics.jsonl", metric)
            write_status(
                run,
                state="running",
                phase="world-model smoke",
                step=offset,
                total_steps=steps,
                message=label,
            )
            if offset == steps or offset % 8 == 0:
                save_checkpoint(
                    run / "checkpoint.pt",
                    {
                        "config": asdict(config),
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": start + offset,
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all()
                        if device == "cuda"
                        else None,
                        "sampling_rng": sampling_rng.get_state(),
                        "batch_size": batch_size,
                        "seed": seed,
                        "data_hash": data_hash,
                        "profile": profile,
                        "model_label": label,
                    },
                )
            print(json.dumps(metric), flush=True)
        atomic_json(
            run / "report.json",
            {
                "quality_gate": "engineering-only",
                "model_label": label,
                "steps": steps,
                "global_step": start + steps,
                "checkpoint_sha256": file_hash(run / "checkpoint.pt"),
                "elapsed_seconds": time.monotonic() - began,
                "limitations": [
                    "bounded smoke; no semantic or survival claim",
                    f"actual batch {batch_size}; paper batch128",
                    "float32 smoke; upstream uses mixed bfloat16",
                ],
            },
        )
        write_status(
            run,
            state="completed",
            phase="world-model smoke",
            step=steps,
            total_steps=steps,
            message="Checkpoint ready for frozen diagnostic fitting",
        )
        return run
    except BaseException as error:
        write_status(
            run,
            state="failed",
            phase="world-model smoke",
            step=offset,
            total_steps=steps,
            message=str(error),
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run-id", default="lewm-mvp")
    parser.add_argument("--profile", choices=["tiny", "reference"], default="reference")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    path = train(
        dataset_root=args.dataset,
        history_root=args.history_root,
        run_id=args.run_id,
        profile=args.profile,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        threads=args.threads,
        resume=args.resume,
        device=args.device,
    )
    print(f"Run artifacts: {path}")


if __name__ == "__main__":
    main()
