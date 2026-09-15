"""Fit the bounded pooling comparison using an existing frozen T4 bank."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path


def validate_result(result: dict, protocol: dict) -> None:
    expected = {
        "world_model_sha256": protocol["inputs"]["world.pt"],
        "data_sha256": protocol["inputs"]["dataset/manifest.json"],
        "frame_index_sha256": protocol["frame_index_sha256"],
        "palette_sha256": protocol["palette_sha256"],
        "core_initial_state_sha256": protocol["core_initial_state_sha256"],
        "milestones": protocol["milestones"],
        "arms": protocol["arms"],
        "world_model_updates": 0,
        "core_parameter_count": 165056,
        "attention_extra_parameter_count": 192,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"Pooling artifact contract mismatch: {key}")


def main():
    import torch

    from dodge_native_game.variants.pixel_repr_ddqn.pooling_probe import run_study
    from dodge_native_game.variants.pixel_repr_ddqn.pooling_readout import (
        make_pooling_decoders,
    )
    from dodge_native_game.variants.pixel_repr_ddqn.probe_bank import _state_digest
    from dodge_native_game.variants.pixel_repr_ddqn.run_artifacts import (
        atomic_json,
        file_hash,
    )

    code = Path("/content/lewm-pooling-code")
    old = Path("/content/lewm-work")
    work = Path("/content/lewm-pooling-work")
    protocol = json.loads((code / "pooling_protocol.json").read_text())
    if protocol["milestones"] != [512, 2048] or protocol["world_model_updates"] != 0:
        raise ValueError("Invalid pooling budget")
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("T4 required")
    for relative, expected in protocol["inputs"].items():
        if file_hash(old / relative) != expected:
            raise ValueError(f"Frozen input changed: {relative}")
    torch.set_num_threads(2)
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/variant_pixel_repr_ddqn"],
        cwd=code,
        check=True,
    )
    # Preserve an independently auditable initial state from this runtime.
    initial = make_pooling_decoders(device="cuda")
    core = initial["mean"].decoder
    if _state_digest(core) != protocol["core_initial_state_sha256"]:
        raise ValueError("Core initialization differs from original spatial study")
    work.mkdir(exist_ok=False)
    history = work / "history"
    history.mkdir()
    torch.save(
        {k: v.cpu() for k, v in initial["mean"].state_dict().items()},
        history / "pooling-initial.pt",
    )
    del initial, core
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    result = run_study(
        old / "dataset",
        old / "world.pt",
        history,
        old / "spatial-banks",
        protocol["run_id"],
        device="cuda",
        milestones=(512, 2048),
    )
    validate_result(result, protocol)
    from dodge_native_game.variants.pixel_repr_ddqn.pooling_gallery import write_gallery

    write_gallery(history, protocol["run_id"], protocol["source_spatial_run"])
    for relative, expected in protocol["inputs"].items():
        if file_hash(old / relative) != expected:
            raise ValueError(f"Frozen input mutated: {relative}")
    atomic_json(
        history / f"{protocol['run_id']}-environment.json",
        {
            "gpu": torch.cuda.get_device_name(0),
            "torch": str(torch.__version__),
            "source_sha256": os.environ["LEWM_SOURCE_HASH"],
            "protocol": protocol,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    )
    archive = Path("/content/lewm-pooling-results.tar.gz")
    with tarfile.open(archive, "w:gz") as out:
        out.add(history, arcname="history")
    Path("/content/lewm-pooling-results.sha256").write_text(file_hash(archive) + "\n")
    print("POOLING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
