"""Frozen-latent survival-cost probe for AD5 action-conditional evaluation.

Read-only world model: encodes dataset windows with a frozen checkpoint and
fits a tiny MLP mapping the projected latent to death-within-H decisions.
The probe is a cost function for 1-step greedy MPC; it never updates LeWM.
Labels come from the recorded terminated flags (train split only for
fitting; validation split only for quoted probe quality).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn

EXPERIMENT: Final[str] = "lewm-survival-probe-v1"
DEATH_HORIZON: Final[int] = 16
PROBE_STEPS: Final[int] = 512
PROBE_BATCH: Final[int] = 256
PROBE_LR: Final[float] = 1e-3
PROBE_SEED: Final[int] = 905
PROBE_FORMAT: Final[str] = "pixel-repr-ddqn-mortal-probe-v1"
PROBE_TRAIN_SEED_START: Final[int] = 22000
PROBE_VALIDATION_SEED_START: Final[int] = 23000
PROBE_PLANNER_SEED: Final[int] = 20260917


class SurvivalProbe(nn.Module):
    """192-dim projected latent -> logit(death within H decisions)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(192, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def collect_mortal_probe_set(
    output: Path,
    *,
    train_episodes: int = 160,
    validation_episodes: int = 80,
    workers: int = 1,
) -> dict[str, Any]:
    """Collect mortal scripted episodes for probe fitting (NOT a corpus).

    Same twenty recipe families as the scale corpus but mortal
    (invulnerable=False) and fresh seeds, so deaths occur.  Variable
    lengths are kept as-is; the frozen training corpora are untouched.
    Probe fitting reads this set; the world model never trains on it.
    """

    import dataclasses

    from .large_practice import (
        SCALE_RECIPE_FAMILIES,
        plan_large_practice,
    )
    from .practice import generate_practice
    from .run_artifacts import atomic_json, file_hash

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"mortal probe output exists: {output}")
    plan = plan_large_practice(
        train_episodes=train_episodes,
        validation_episodes=validation_episodes,
        train_seed_start=PROBE_TRAIN_SEED_START,
        validation_seed_start=PROBE_VALIDATION_SEED_START,
        planner_seed=PROBE_PLANNER_SEED,
        families=SCALE_RECIPE_FAMILIES,
    )
    episodes: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for spec in plan:
        config = dataclasses.replace(
            spec.config, invulnerable=False, name=f"mortal-{spec.split}"
        )
        episode_dir = output / "episodes" / spec.split / spec.recipe_id
        record = generate_practice(config, episode_dir, seed=spec.seed)
        archive = episode_dir / "episode.npz"
        episodes[spec.split].append(
            {
                "episode_id": spec.recipe_id,
                "recipe_family": spec.recipe_family,
                "seed": spec.seed,
                "path": archive.relative_to(output).as_posix(),
                "sha256": file_hash(archive),
                "decisions": int(record["decision_count"]),
                "terminated": bool(record["terminated"]),
            }
        )
    manifest = {
        "format": PROBE_FORMAT,
        "splits": {split: len(rows) for split, rows in episodes.items()},
        "deaths": {
            split: sum(1 for row in rows if row["terminated"])
            for split, rows in episodes.items()
        },
        "episodes": episodes,
    }
    atomic_json(output / "manifest.json", manifest)
    (output / "READY").write_text(PROBE_FORMAT + "\n")
    return manifest


def _windows_with_labels(
    probe_root: Path,
    split: str,
    *,
    max_windows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Collect (window, death-label) pairs from a mortal probe set."""

    import json

    probe_root = Path(probe_root)
    manifest = json.loads((probe_root / "manifest.json").read_text())
    if manifest.get("format") != PROBE_FORMAT:
        raise ValueError("not a mortal probe set")
    entries = manifest["episodes"][split]
    selector = np.random.default_rng(seed)
    order = selector.permutation(len(entries))
    windows: list[np.ndarray] = []
    labels: list[int] = []
    provenance: list[str] = []
    for entry_index in order:
        entry = entries[int(entry_index)]
        with np.load(probe_root / entry["path"], allow_pickle=False) as payload:
            pixels = np.asarray(payload["pixels"])
            terminated = np.asarray(payload["terminated"]).astype(bool)
        count = len(terminated)
        for start in range(0, count - 3):
            end_frame = start + 3
            horizon = terminated[end_frame : end_frame + DEATH_HORIZON]
            labels.append(int(bool(horizon.any())) if len(horizon) else 0)
            windows.append(pixels[start : start + 4].copy())
            provenance.append(f"{entry['episode_id']}:{start}")
            if len(windows) >= max_windows:
                break
        if len(windows) >= max_windows:
            break
        del pixels
    return (
        np.stack(windows).astype(np.uint8),
        np.asarray(labels, dtype=np.float32),
        provenance,
    )


def _encode_latents(
    model: nn.Module,
    windows: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Projected CLS latents for uint8 windows, no gradients."""

    latents: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for index in range(0, len(windows), batch_size):
            pixels = torch.from_numpy(windows[index : index + batch_size]).to(device)
            latents.append(model.encode(pixels)[:, -1, :].detach().cpu())
    return torch.cat(latents)


def fit_probe(
    probe_root: Path,
    checkpoint: Path,
    output: Path,
    *,
    device: str = "cpu",
    steps: int = PROBE_STEPS,
    batch_size: int = PROBE_BATCH,
    seed: int = PROBE_SEED,
    max_windows: int = 20000,
) -> dict[str, Any]:
    """Fit the survival probe on mortal train windows; quote on mortal val."""

    from .pretrain import load_model
    from .run_artifacts import atomic_json, file_hash

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    device_obj = torch.device(device)
    world_hash = file_hash(checkpoint)
    model, _ = load_model(checkpoint)
    model = model.to(device_obj).eval()
    model.requires_grad_(False)

    train_pixels, train_labels, _ = _windows_with_labels(
        probe_root, "train", max_windows=max_windows, seed=seed
    )
    val_pixels, val_labels, _ = _windows_with_labels(
        probe_root, "validation", max_windows=max_windows // 4, seed=seed + 1
    )
    train_z = _encode_latents(model, train_pixels, device=device_obj)
    val_z = _encode_latents(model, val_pixels, device=device_obj)
    train_y = torch.from_numpy(train_labels)
    val_y = torch.from_numpy(val_labels)
    del train_pixels, val_pixels

    probe = SurvivalProbe()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=PROBE_LR)
    criterion = nn.BCEWithLogitsLoss()
    sampler = torch.Generator().manual_seed(seed)
    began = time.monotonic()
    initial_loss: float | None = None
    for _ in range(steps):
        indices = torch.randint(len(train_z), (batch_size,), generator=sampler)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(probe(train_z[indices]), train_y[indices])
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        optimizer.step()
    assert initial_loss is not None
    probe.eval()
    with torch.no_grad():
        train_loss = float(criterion(probe(train_z), train_y))
        val_loss = float(criterion(probe(val_z), val_y))
        val_prob = torch.sigmoid(probe(val_z))
        order = torch.argsort(val_prob)
        ranked = val_y[order]
        positives = int(val_y.sum().item())
        if positives and positives < len(val_y):
            retrieved = torch.cumsum(ranked.flip(0), 0).float()
            precision = retrieved / torch.arange(1, len(val_y) + 1).flip(0).float()
            val_auprc = float(
                (precision * ranked.flip(0)).sum() / positives
            )
        else:
            val_auprc = None
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": probe.state_dict(), "experiment": EXPERIMENT,
         "world_model_sha256": world_hash},
        output,
    )
    report = {
        "experiment": EXPERIMENT,
        "world_model_sha256": world_hash,
        "probe_set": str(probe_root),
        "probe_manifest_sha256": file_hash(Path(probe_root) / "manifest.json"),
        "device": device_obj.type,
        "steps": steps,
        "batch_size": batch_size,
        "seed": seed,
        "death_horizon": DEATH_HORIZON,
        "train_windows": len(train_z),
        "val_windows": len(val_z),
        "train_positive_rate": float(train_y.mean()),
        "val_positive_rate": float(val_y.mean()),
        "initial_loss": initial_loss,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_auprc": val_auprc,
        "probe_sha256": file_hash(output),
        "elapsed_seconds": time.monotonic() - began,
    }
    atomic_json(output.with_suffix(".json"), report)
    auprc_str = f"{val_auprc:.4f}" if val_auprc is not None else "null"
    print(
        f"PROBE_COMPLETE train_loss={train_loss:.4f} "
        f"val_loss={val_loss:.4f} val_auprc={auprc_str} "
        f"pos_rate={float(val_y.mean()):.3f}",
        flush=True,
    )
    return report


__all__ = [
    "DEATH_HORIZON",
    "EXPERIMENT",
    "PROBE_BATCH",
    "PROBE_FORMAT",
    "PROBE_LR",
    "PROBE_PLANNER_SEED",
    "PROBE_SEED",
    "PROBE_STEPS",
    "PROBE_TRAIN_SEED_START",
    "PROBE_VALIDATION_SEED_START",
    "SurvivalProbe",
    "collect_mortal_probe_set",
    "fit_probe",
]
