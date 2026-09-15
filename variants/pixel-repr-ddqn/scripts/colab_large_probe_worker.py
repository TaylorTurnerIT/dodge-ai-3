"""Extract frozen representations, fit matched heads, and package T4 evidence."""


def main():
    import json
    import os
    import shutil
    import subprocess
    import sys
    import tarfile
    from pathlib import Path

    import torch

    from dodge_native_game.variants.pixel_repr_ddqn.large_probe import run_study
    from dodge_native_game.variants.pixel_repr_ddqn.run_artifacts import file_hash

    root = Path.cwd()
    protocol = json.loads((root / "large_probe_protocol.json").read_text())
    run_id = os.environ["LEWM_RUN_ID"]
    assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0)
    assert file_hash(root / "checkpoint.pt") == protocol["checkpoint_sha256"]
    assert file_hash(root / "dataset/manifest.json") == protocol["data_hash"]
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/variant_pixel_repr_ddqn", "-q"],
        check=True,
    )
    run_study(
        root / "checkpoint.pt",
        root / "dataset",
        root / "history",
        run_id,
        milestones=tuple(protocol["milestones"]),
        batch_size=protocol["batch_size"],
        device="cuda",
        bank_root=root / "probe-banks",
    )
    assert file_hash(root / "checkpoint.pt") == protocol["checkpoint_sha256"]
    provenance = root / "history" / f"{run_id}-bank-provenance"
    for split in ("train", "validation"):
        destination = provenance / split
        destination.mkdir(parents=True)
        for filename in ("metadata.json", "index.json", "READY"):
            shutil.copy2(
                root / "probe-banks" / split / filename, destination / filename
            )
    environment = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "source_sha256": os.environ["LEWM_SOURCE_HASH"],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "world_model_updates": 0,
        "new_game_steps": 0,
        "protocol": protocol,
    }
    (root / "history" / f"{run_id}-environment.json").write_text(
        json.dumps(environment, indent=2)
    )
    with tarfile.open("/content/lewm-large-probe-results.tar.gz", "w:gz") as output:
        output.add(root / "history", arcname="history")
    print("LEWM_LARGE_PROBE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
