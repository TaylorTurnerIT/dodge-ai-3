"""Frozen model, separate diagnostic fitting and evaluation on Colab T4."""


def main():
    import json
    import os
    import subprocess
    import sys
    import tarfile
    from pathlib import Path

    import torch

    from dodge_native_game.variants.pixel_repr_ddqn.decoder_study import run_study
    from dodge_native_game.variants.pixel_repr_ddqn.run_artifacts import file_hash

    root = Path.cwd()
    assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0)
    protocol = json.loads((root / "decoder_protocol.json").read_text())
    assert file_hash(root / "checkpoint.pt") == protocol["checkpoint_sha256"]
    assert file_hash(root / "dataset/manifest.json") == protocol["data_hash"]
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/variant_pixel_repr_ddqn", "-q"],
        check=True,
    )
    resume = root / "resume_decoder.pt" if protocol["resume_decoder_sha256"] else None
    if resume is not None:
        assert file_hash(resume) == protocol["resume_decoder_sha256"]
    run_study(
        root / "checkpoint.pt",
        root / "dataset",
        root / "history",
        os.environ["LEWM_RUN_ID"],
        steps=protocol["decoder_total_steps"],
        resume_decoder=resume,
    )
    assert file_hash(root / "checkpoint.pt") == protocol["checkpoint_sha256"]
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
    (root / "history/remote_environment.json").write_text(
        json.dumps(environment, indent=2)
    )
    with tarfile.open("/content/lewm-decoder-results.tar.gz", "w:gz") as output:
        output.add(root / "history", arcname="history")
    print("LEWM_DECODER_STUDY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
