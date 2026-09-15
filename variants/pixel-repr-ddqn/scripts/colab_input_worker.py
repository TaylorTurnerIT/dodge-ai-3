"""Fixed phase order: paired world fitting, banks, frozen probes, evidence archive."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path


def main():
    import torch

    from dodge_native_game.variants.pixel_repr_ddqn.input_pretrain import run_pair
    from dodge_native_game.variants.pixel_repr_ddqn.input_probe import run_probes
    from dodge_native_game.variants.pixel_repr_ddqn.run_artifacts import (
        atomic_json,
        file_hash,
    )

    root = Path("/content/lewm-work")
    protocol = json.loads((root / "input_protocol.json").read_text())
    run = protocol["run_id"]
    expected = {
        "experiment": "lewm-input-encoding-v1",
        "world_steps": 1024,
        "world_milestones": [512, 1024],
        "world_batch_size": 32,
        "world_init_seed": 42,
        "world_sampling_seed": 43,
        "world_stochastic_seed": 44,
        "decoder_milestones": [512, 2048, 8192],
        "decoder_init_seed": 904,
        "decoder_sampling_seed": 903,
        "loss_kind": "palette-ce",
        "representation": "cls",
        "arms": ["rgb", "palette"],
        "worker_timeout_seconds": 7200,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Input protocol mismatch: {key}")
    assert file_hash(root / "dataset/manifest.json") == protocol["data_sha256"]
    assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0)
    torch.set_num_threads(2)
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/variant_pixel_repr_ddqn"],
        cwd=root,
        check=True,
    )
    torch.cuda.reset_peak_memory_stats()
    print("INPUT_PHASE world fitting", flush=True)
    result = run_pair(root / "dataset", root / "history", run_id=run, device="cuda")
    checkpoints = {arm: Path(path) for arm, path in result["checkpoints"].items()}
    print("INPUT_PHASE feature extraction then frozen probes", flush=True)
    run_probes(
        root / "dataset",
        checkpoints,
        root / "history",
        root / "input-banks",
        run,
        device="cuda",
    )
    environment = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "source_sha256": os.environ["LEWM_SOURCE_HASH"],
        "protocol": protocol,
    }
    atomic_json(root / "history" / f"{run}-environment.json", environment)
    archive = Path("/content/lewm-input-results.tar.gz")
    with tarfile.open(archive, "w:gz") as output:
        output.add(root / "history", arcname="history")
        for arm in ("rgb", "palette"):
            for split in ("train", "validation"):
                for name in ("metadata.json", "index.json", "READY"):
                    output.add(
                        root / "input-banks" / arm / split / name,
                        arcname=f"history/{run}-bank-provenance/{arm}/{split}/{name}",
                    )
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    Path("/content/lewm-input-results.sha256").write_text(digest + "\n")
    print("INPUT_COMPLETE", digest, archive.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
