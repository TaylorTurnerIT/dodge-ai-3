"""Spatial sidecars aligned to the existing frozen frame-bank contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .probe_bank import _state_digest
from .run_artifacts import atomic_json, file_hash
from .spatial_readout import broadcast_cls, pixel_patch_features


class ExpandedFeatures:
    """Lazy controls: avoid storing repeated CLS or lossless pixel patches."""

    def __init__(self, source, *, palette=None):
        self.source = source
        self.palette = palette

    def __len__(self):
        return len(self.source)

    def __getitem__(self, indices):
        values = np.asarray(self.source[indices])
        single = isinstance(indices, (int, np.integer))
        if single:
            values = values[None]
        tensor = torch.from_numpy(values.copy())
        result = (
            broadcast_cls(tensor)
            if self.palette is None
            else pixel_patch_features(tensor, self.palette)
        )
        result = result.cpu().numpy()
        return result[0] if single else result


def extract_patches(model, bank, root: Path, *, device: str, batch_size: int = 32):
    """Publish a sidecar only after exact provenance and frozen-state checks."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    before = _state_digest(model)
    for key in ("pixels", "cls"):
        if file_hash(bank.root / f"{key}.npy") != bank.metadata["files"][key]["sha256"]:
            raise ValueError(f"Corrupt source bank: {key}")
    patches = np.lib.format.open_memmap(
        root / "patches.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(bank), 256, 192),
    )
    with torch.inference_mode():
        for start in range(0, len(bank), batch_size):
            stop = min(start + batch_size, len(bank))
            inputs = (
                torch.from_numpy(np.array(bank.pixels[start:stop]))
                .unsqueeze(1)
                .to(device)
            )
            cls, local = model.encode_readout_tokens(inputs)
            cls, local = cls[:, 0].cpu().numpy(), local[:, 0].cpu().numpy()
            if local.shape != (stop - start, 256, 192) or not np.isfinite(local).all():
                raise ValueError("Invalid extracted spatial tokens")
            if not np.allclose(cls, bank.cls[start:stop], atol=2e-5, rtol=2e-5):
                raise ValueError("Spatial extraction CLS differs from source bank")
            patches[start:stop] = local
    patches.flush()
    if _state_digest(model) != before:
        raise RuntimeError("World model changed during spatial extraction")
    metadata = {
        "format": "lewm-spatial-sidecar-v1",
        "source_metadata_sha256": file_hash(bank.root / "metadata.json"),
        "source_index_sha256": file_hash(bank.root / "index.json"),
        "model_state_sha256": before,
        "patches_sha256": file_hash(root / "patches.npy"),
        "shape": list(patches.shape),
        "dtype": str(patches.dtype),
        "order": "row-major grid, final encoder layer, CLS excluded",
    }
    atomic_json(root / "metadata.json", metadata)
    (root / "READY").write_text("lewm-spatial-sidecar-v1\n")
    del patches
    return open_patches(bank, root)


def open_patches(bank, root: Path):
    root = Path(root)
    if (root / "READY").read_text() != "lewm-spatial-sidecar-v1\n":
        raise ValueError("Spatial bank is not ready")
    metadata = json.loads((root / "metadata.json").read_text())
    expected = {
        "format": "lewm-spatial-sidecar-v1",
        "source_metadata_sha256": file_hash(bank.root / "metadata.json"),
        "source_index_sha256": file_hash(bank.root / "index.json"),
        "patches_sha256": file_hash(root / "patches.npy"),
        "shape": [len(bank), 256, 192],
        "dtype": "float32",
    }
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("Spatial sidecar provenance mismatch")
    result = np.load(root / "patches.npy", mmap_mode="r", allow_pickle=False)
    if list(result.shape) != expected["shape"] or result.dtype != np.float32:
        raise ValueError("Spatial sidecar array mismatch")
    return result
