"""Single-arm palette continuation worker for §AD scaled training.

Fixed phase order: setup is done by the remote driver; this worker verifies
the frozen protocol inputs, restores the base checkpoint with full RNG
state, runs ``extra_steps`` more updates via ``input_pretrain.continue_run``,
then packages history plus the new checkpoint into a verified archive.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path


def main() -> None:
    import torch

    from dodge_native_game.variants.pixel_repr_ddqn.input_pretrain import (
        continue_run,
    )
    from dodge_native_game.variants.pixel_repr_ddqn.run_artifacts import (
        atomic_json,
        file_hash,
    )

    root = Path("/content/lewm-scale-work")
    protocol = json.loads((root / "scale_protocol.json").read_text())
    run = protocol["run_id"]
    expected = {
        "experiment": "lewm-scale-continuation-v1",
        "input_arm": "palette",
        "world_batch_size": 32,
        "world_init_seed": 42,
        "world_sampling_seed": 43,
        "world_stochastic_seed": 44,
        "worker_timeout_seconds": 7200,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Scale protocol mismatch: {key}")
    extra_steps = int(protocol["extra_steps"])
    base_step = int(protocol["base_step"])
    if extra_steps < 1 or base_step < 1:
        raise ValueError("Scale protocol steps must be positive")
    assert file_hash(root / "dataset/manifest.json") == protocol["data_sha256"]
    base_checkpoint = root / "base" / "checkpoint.pt"
    assert file_hash(base_checkpoint) == protocol["base_checkpoint_sha256"]
    assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0)
    torch.set_num_threads(2)
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/variant_pixel_repr_ddqn"],
        cwd=root / "code",
        check=True,
    )
    torch.cuda.reset_peak_memory_stats()
    print("SCALE_PHASE continuation fitting", flush=True)
    result = continue_run(
        base_checkpoint,
        root / "dataset",
        root / "history",
        run,
        extra_steps=extra_steps,
        checkpoint_every=int(protocol["checkpoint_every"]),
        device="cuda",
    )
    checkpoint = Path(result["checkpoint"])  # type: ignore[arg-type]
    manifest = json.loads(Path(result["manifest"]).read_text())  # type: ignore[arg-type]
    total_steps = base_step + extra_steps
    if manifest.get("steps") != total_steps:
        raise ValueError("Continuation manifest step mismatch")
    if manifest.get("base_checkpoint_sha256") != protocol["base_checkpoint_sha256"]:
        raise ValueError("Continuation manifest base mismatch")
    for relative, expected_hash in protocol["inputs"].items():
        if file_hash(root / relative) != expected_hash:
            raise ValueError(f"Frozen input mutated: {relative}")
    environment = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "source_sha256": os.environ["LEWM_SOURCE_HASH"],
        "protocol": protocol,
    }
    atomic_json(root / "history" / f"{run}-environment.json", environment)
    archive = Path("/content/lewm-scale-results.tar.gz")
    with tarfile.open(archive, "w:gz") as output:
        output.add(root / "history", arcname="history")
        output.add(checkpoint, arcname=f"history/{run}/checkpoint.pt")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    Path("/content/lewm-scale-results.sha256").write_text(digest + "\n")
    print("SCALE_COMPLETE", digest, archive.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
