"""Fresh-process checks, collection, training and frozen diagnostics on Colab."""


def main():
    import json
    import os
    import subprocess
    import sys
    import tarfile
    from pathlib import Path

    import torch

    root = Path.cwd()
    RUN_ID = os.environ["LEWM_RUN_ID"]
    SOURCE_HASH = os.environ["LEWM_SOURCE_HASH"]
    assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0)
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/variant_pixel_repr_ddqn", "-q"],
        check=True,
    )
    from dodge_native_game.variants.pixel_repr_ddqn.collect import collect_dataset
    from dodge_native_game.variants.pixel_repr_ddqn.pretrain import train
    from dodge_native_game.variants.pixel_repr_ddqn.probe import fit_probe

    dataset = root / "dataset"
    collect_dataset(
        dataset,
        train_seeds=[11, 12, 13],
        validation_seeds=[10011],
        max_steps_per_episode=64,
        seed=42,
    )
    run = train(
        dataset_root=dataset,
        history_root=root / "history",
        run_id=RUN_ID,
        profile="reference",
        steps=32,
        batch_size=4,
        device="cuda",
    )
    fit_probe(run, dataset, steps=32, batch_size=4, device="cuda")
    (run / "remote_environment.json").write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "source_sha256": SOURCE_HASH,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            },
            indent=2,
        )
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dodge_native_game.variants.cnn_image_ddqn.run",
            "--history-root",
            str(root / "legacy"),
            "--run-id",
            RUN_ID + "-legacy",
            "--steps",
            "32",
            "--stack-size",
            "1",
            "--device",
            "cuda",
        ],
        check=True,
    )
    with tarfile.open("/content/lewm-results.tar.gz", "w:gz") as output:
        for name in ["history", "dataset", "legacy"]:
            output.add(root / name, arcname=name)
    print("LEWM_MVP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
