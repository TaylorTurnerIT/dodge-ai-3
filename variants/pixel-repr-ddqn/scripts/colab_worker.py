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
    protocol_path = root / "practice_protocol.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else None
    if protocol is None:
        collect_dataset(
            dataset,
            train_seeds=[11, 12, 13],
            validation_seeds=[10011],
            max_steps_per_episode=64,
            seed=42,
            scenario=root / "scenario.toml"
            if (root / "scenario.toml").is_file()
            else None,
        )
    else:
        from dodge_native_game.variants.pixel_repr_ddqn.dataset import (
            PixelSequenceDataset,
        )
        from dodge_native_game.variants.pixel_repr_ddqn.run_artifacts import file_hash

        assert file_hash(dataset / "manifest.json") == protocol["data_hash"]
        for split in ("train", "validation"):
            assert len(PixelSequenceDataset(dataset, split=split)) > 0
    run = train(
        dataset_root=dataset,
        history_root=root / "history",
        run_id=RUN_ID,
        profile="reference",
        steps=protocol["model_updates"] if protocol else 32,
        batch_size=protocol["batch_size"] if protocol else 4,
        experiment=protocol["experiment"] if protocol else "mvp",
        device="cuda",
    )
    fit_probe(
        run,
        dataset,
        steps=protocol["decoder_updates"] if protocol else 32,
        batch_size=protocol["batch_size"] if protocol else 4,
        device="cuda",
    )
    if protocol:
        from dodge_native_game.variants.pixel_repr_ddqn.dynamics import (
            evaluate_dynamics,
        )
        from dodge_native_game.variants.pixel_repr_ddqn.pretrain import load_model
        from dodge_native_game.variants.pixel_repr_ddqn.run_artifacts import atomic_json

        model, _ = load_model(run / "checkpoint.pt")
        model = model.to("cuda")
        checks = {
            split: evaluate_dynamics(
                model, PixelSequenceDataset(dataset, split=split), "cuda"
            )
            for split in ("train", "validation")
        }
        checks.update(
            checkpoint_sha256=file_hash(run / "checkpoint.pt"), protocol=protocol
        )
        atomic_json(run / "dynamics.json", checks)
    if protocol and protocol.get("calibrate_encoder"):
        from dodge_native_game.variants.pixel_repr_ddqn.calibration import (
            export_calibrated_run,
        )

        calibrated = export_calibrated_run(
            run, dataset, root / "history", RUN_ID + "-calibrated"
        )
        calibrated_model, _ = load_model(calibrated / "checkpoint.pt")
        calibrated_model = calibrated_model.to("cuda")
        calibrated_checks = {
            split: evaluate_dynamics(
                calibrated_model, PixelSequenceDataset(dataset, split=split), "cuda"
            )
            for split in ("train", "validation")
        }
        calibrated_checks.update(
            checkpoint_sha256=file_hash(calibrated / "checkpoint.pt"), protocol=protocol
        )
        atomic_json(calibrated / "dynamics.json", calibrated_checks)
        del calibrated_model
        fit_probe(calibrated, dataset, steps=256, batch_size=8, device="cuda")
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
