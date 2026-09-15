"""Run the frozen local spatial readout protocol on a Colab T4."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

EXPERIMENT = "lewm-spatial-readout-v1"
ARMS = ("cls", "patch", "pixels")
MILESTONES = (512, 2048, 8192)


def _require_protocol(protocol: dict[str, Any]) -> None:
    expected = {
        "experiment": EXPERIMENT,
        "checkpoint_name": "world.pt",
        "world_updates": 0,
        "world_model_updates": 0,
        "milestones": list(MILESTONES),
        "batch_size": 32,
        "decoder_init_seed": 904,
        "decoder_sampling_seed": 903,
        "loss_kind": "palette-ce",
        "decoder_kind": "local-patch",
        "arms": list(ARMS),
        "worker_timeout_seconds": 7200,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Spatial protocol mismatch: {key}")
    if not isinstance(protocol.get("run_id"), str) or not protocol["run_id"]:
        raise ValueError("Spatial protocol run_id is invalid")
    for key in ("checkpoint_sha256", "data_sha256"):
        value = protocol.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Spatial protocol {key} is invalid")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")


def _bank_artifacts(root: Path):
    bank_root = root / "spatial-banks"
    required = []
    for kind, names in (
        ("standard", ("metadata.json", "index.json", "READY")),
        ("spatial", ("metadata.json", "READY")),
    ):
        for split in ("train", "validation"):
            required.extend(bank_root / kind / split / name for name in names)
    required.extend(
        (
            bank_root / "spatial" / "validation" / "patches.npy",
            bank_root / "standard" / "validation" / "cls.npy",
        )
    )
    return bank_root, tuple(required)


def main() -> None:
    import torch

    from dodge_native_game.variants.pixel_repr_ddqn.run_artifacts import (
        atomic_json,
        file_hash,
    )
    from dodge_native_game.variants.pixel_repr_ddqn.spatial_gallery import write_gallery
    from dodge_native_game.variants.pixel_repr_ddqn.spatial_probe import run_study

    root = Path("/content/lewm-work")
    protocol = json.loads((root / "spatial_protocol.json").read_text())
    if not isinstance(protocol, dict):
        raise ValueError("Spatial protocol must contain a JSON object")
    _require_protocol(protocol)
    run_id = protocol["run_id"]
    if os.environ.get("LEWM_RUN_ID") != run_id:
        raise ValueError("worker run ID does not match spatial protocol")
    checkpoint = root / str(protocol["checkpoint_name"])
    dataset_manifest = root / "dataset" / "manifest.json"
    _require_file(checkpoint, "world checkpoint")
    _require_file(dataset_manifest, "dataset manifest")
    if file_hash(checkpoint) != protocol["checkpoint_sha256"]:
        raise RuntimeError("world checkpoint SHA-256 does not match protocol")
    if file_hash(dataset_manifest) != protocol["data_sha256"]:
        raise RuntimeError("dataset manifest SHA-256 does not match protocol")
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("a Colab Tesla T4 is required")
    torch.set_num_threads(2)
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/variant_pixel_repr_ddqn"],
        cwd=root,
        check=True,
    )
    torch.cuda.reset_peak_memory_stats()
    (root / "history").mkdir(parents=True, exist_ok=False)
    print("SPATIAL_PHASE frozen diagnostic study", flush=True)
    run_study(
        root / "dataset",
        checkpoint,
        root / "history",
        root / "spatial-banks",
        run_id,
        device="cuda",
        milestones=tuple(protocol["milestones"]),
    )
    write_gallery(root / "history", run_id)
    if file_hash(checkpoint) != protocol["checkpoint_sha256"]:
        raise RuntimeError("world checkpoint changed during spatial study")
    comparison = root / "history" / f"{run_id}-spatial-comparison.json"
    _require_file(comparison, "spatial comparison")
    bank_root, required = _bank_artifacts(root)
    for path in required:
        _require_file(path, "spatial bank artifact")
    provenance = root / "history" / f"{run_id}-bank-provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    # Keep the immutable input bundle and protocol beside the compact bank audit.
    # The run directories already contain hard-linked copies of this checkpoint.
    import shutil

    shutil.copy2(checkpoint, provenance / "world.pt")
    shutil.copy2(root / "spatial_protocol.json", provenance / "spatial_protocol.json")
    environment = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "source_sha256": os.environ["LEWM_SOURCE_HASH"],
        "protocol": protocol,
        "world_checkpoint_sha256": file_hash(checkpoint),
    }
    atomic_json(root / "history" / f"{run_id}-environment.json", environment)
    archive = Path("/content/lewm-spatial-results.tar.gz")
    with tarfile.open(archive, "w:gz") as output:
        output.add(root / "history", arcname="history")
        for kind, names in (
            ("standard", ("metadata.json", "index.json", "READY")),
            ("spatial", ("metadata.json", "READY")),
        ):
            for split in ("train", "validation"):
                for name in names:
                    source = bank_root / kind / split / name
                    output.add(
                        source,
                        arcname=f"history/{run_id}-bank-provenance/{kind}/{split}/{name}",
                    )
        for kind, split, name in (
            ("spatial", "validation", "patches.npy"),
            ("standard", "validation", "cls.npy"),
        ):
            source = bank_root / kind / split / name
            output.add(
                source,
                arcname=f"history/{run_id}-bank-provenance/{kind}/{split}/{name}",
            )
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    Path("/content/lewm-spatial-results.sha256").write_text(digest.hexdigest() + "\n")
    print("SPATIAL_COMPLETE", digest.hexdigest(), archive.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
