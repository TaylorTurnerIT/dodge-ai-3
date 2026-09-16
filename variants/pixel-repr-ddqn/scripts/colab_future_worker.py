"""Run the frozen predicted next-frame decode study on a Colab T4."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

EXPERIMENT = "lewm-future-decode-v1"
MILESTONES = (512, 2048)


READOUTS = {
    "predicted-latent-broadcast": "predicted",
    "actual-next-latent": "actual",
}


def validate_result(result: dict, protocol: dict) -> None:
    expected = {
        "experiment": EXPERIMENT,
        "world_model_sha256": protocol["checkpoint_sha256"],
        "data_sha256": protocol["data_sha256"],
        "current_decoder_sha256": protocol["current_decoder_sha256"],
        "milestones": protocol["milestones"],
        "world_model_updates": 0,
        "batch_size": 32,
        "parameter_count": 165056,
        "scene_count": 16,
        "loss_kind": "balanced-bright",
        "equal_class_weights": True,
        "readout": protocol["readout"],
    }
    if protocol["readout"] == "actual-next-latent":
        expected["ab_source_run"] = protocol["ab_source_run"]
        expected["ab_decoder_sha256"] = protocol["ab_decoder_sha256"]
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"Future artifact contract mismatch: {key}")


def _require_protocol(protocol: dict) -> None:
    expected = {
        "experiment": EXPERIMENT,
        "checkpoint_name": "world.pt",
        "current_decoder_name": "current-decoder.pt",
        "world_updates": 0,
        "world_model_updates": 0,
        "milestones": list(MILESTONES),
        "batch_size": 32,
        "decoder_init_seed": 904,
        "decoder_sampling_seed": 903,
        "loss_kind": "balanced-bright",
        "equal_class_weights": True,
        "scene_count": 16,
        "worker_timeout_seconds": 7200,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Future protocol mismatch: {key}")
    if protocol.get("readout") not in READOUTS:
        raise ValueError("Future protocol readout is invalid")
    if not isinstance(protocol.get("run_id"), str) or not protocol["run_id"]:
        raise ValueError("Future protocol run_id is invalid")
    keys = ["checkpoint_sha256", "current_decoder_sha256", "data_sha256"]
    if protocol["readout"] == "actual-next-latent":
        if protocol.get("ab_decoder_name") != "ab-decoder.pt":
            raise ValueError("Future protocol ab_decoder_name is invalid")
        if (
            not isinstance(protocol.get("ab_source_run"), str)
            or not protocol["ab_source_run"]
        ):
            raise ValueError("Future protocol ab_source_run is invalid")
        keys.append("ab_decoder_sha256")
    for key in keys:
        value = protocol.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Future protocol {key} is invalid")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")


def main() -> None:
    import torch

    from dodge_native_game.variants.pixel_repr_ddqn.future_decode import run_study
    from dodge_native_game.variants.pixel_repr_ddqn.run_artifacts import (
        atomic_json,
        file_hash,
    )

    root = Path("/content/lewm-work")
    protocol = json.loads((root / "future_protocol.json").read_text())
    if not isinstance(protocol, dict):
        raise ValueError("Future protocol must contain a JSON object")
    _require_protocol(protocol)
    run_id = protocol["run_id"]
    if os.environ.get("LEWM_RUN_ID") != run_id:
        raise ValueError("worker run ID does not match future protocol")
    checkpoint = root / str(protocol["checkpoint_name"])
    current_decoder = root / str(protocol["current_decoder_name"])
    dataset_manifest = root / "dataset" / "manifest.json"
    _require_file(checkpoint, "world checkpoint")
    _require_file(current_decoder, "current-frame decoder")
    _require_file(dataset_manifest, "dataset manifest")
    if file_hash(checkpoint) != protocol["checkpoint_sha256"]:
        raise RuntimeError("world checkpoint SHA-256 does not match protocol")
    if file_hash(current_decoder) != protocol["current_decoder_sha256"]:
        raise RuntimeError("current-frame decoder SHA-256 does not match protocol")
    if file_hash(dataset_manifest) != protocol["data_sha256"]:
        raise RuntimeError("dataset manifest SHA-256 does not match protocol")
    ab_decoder = None
    if protocol["readout"] == "actual-next-latent":
        ab_decoder = root / str(protocol["ab_decoder_name"])
        _require_file(ab_decoder, "reference predicted readout")
        if file_hash(ab_decoder) != protocol["ab_decoder_sha256"]:
            raise RuntimeError("reference readout SHA-256 does not match protocol")
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("a Colab Tesla T4 is required")
    torch.set_num_threads(2)
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/variant_pixel_repr_ddqn"],
        cwd=root,
        check=True,
    )
    torch.cuda.reset_peak_memory_stats()
    (root / "history").mkdir(parents=True, exist_ok=True)
    print("FUTURE_PHASE frozen diagnostic study", flush=True)
    result = run_study(
        root / "dataset",
        checkpoint,
        current_decoder,
        root / "history",
        run_id,
        device="cuda",
        milestones=tuple(protocol["milestones"]),
        readout=READOUTS[protocol["readout"]],
        ab_decoder=ab_decoder,
        ab_source_run=protocol.get("ab_source_run"),
    )
    if file_hash(checkpoint) != protocol["checkpoint_sha256"]:
        raise RuntimeError("world checkpoint changed during future study")
    if file_hash(current_decoder) != protocol["current_decoder_sha256"]:
        raise RuntimeError("current-frame decoder changed during future study")
    if (
        ab_decoder is not None
        and file_hash(ab_decoder) != protocol["ab_decoder_sha256"]
    ):
        raise RuntimeError("reference readout changed during future study")
    validate_result(result, protocol)
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
    archive = Path("/content/lewm-future-results.tar.gz")
    with tarfile.open(archive, "w:gz") as output:
        output.add(root / "history", arcname="history")
        output.add(
            root / "future_protocol.json",
            arcname=f"history/{run_id}-future-protocol.json",
        )
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    Path("/content/lewm-future-results.sha256").write_text(digest.hexdigest() + "\n")
    print("FUTURE_COMPLETE", digest.hexdigest(), archive.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
