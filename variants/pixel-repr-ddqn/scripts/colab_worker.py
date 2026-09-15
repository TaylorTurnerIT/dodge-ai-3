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
    from dodge_native_game.variants.pixel_repr_ddqn.run_artifacts import (
        atomic_json,
        create_run,
    )

    runs = [run]
    if protocol and protocol["experiment"] == "practice-batch32-v1":
        import shutil

        manifest = json.loads((run / "manifest.json").read_text())
        manifest.update(total_steps=128, retained_from=RUN_ID)
        retained = create_run(root / "history", RUN_ID + "-step128", manifest)
        shutil.copyfile(run / "checkpoint-128.pt", retained / "checkpoint.pt")
        shutil.copyfile(run / "config.json", retained / "config.json")
        rows = [
            json.loads(line)
            for line in (run / "metrics.jsonl").read_text().splitlines()
        ]
        (retained / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows if row["step"] <= 128)
        )
        atomic_json(
            retained / "report.json",
            {
                "quality_gate": "engineering-only",
                "steps": 128,
                "checkpoint_sha256": file_hash(retained / "checkpoint.pt"),
                "comparison": "4096 windows; fewer optimizer updates than batch8",
            },
        )
        runs.insert(0, retained)

    def diagnose(target):
        fit_probe(
            target,
            dataset,
            steps=protocol["decoder_updates"] if protocol else 32,
            batch_size=8 if protocol else 4,
            device="cuda",
        )
        if protocol:
            from dodge_native_game.variants.pixel_repr_ddqn.dynamics import (
                evaluate_dynamics,
            )
            from dodge_native_game.variants.pixel_repr_ddqn.pretrain import load_model

            model, _ = load_model(target / "checkpoint.pt")
            model = model.to("cuda")
            checks = {
                split: evaluate_dynamics(
                    model, PixelSequenceDataset(dataset, split=split), "cuda"
                )
                for split in ("train", "validation")
            }
            checks.update(
                checkpoint_sha256=file_hash(target / "checkpoint.pt"), protocol=protocol
            )
            atomic_json(target / "dynamics.json", checks)

    for target in runs:
        diagnose(target)
        if protocol and protocol.get("calibrate_encoder"):
            from dodge_native_game.variants.pixel_repr_ddqn.calibration import (
                export_calibrated_run,
            )

            calibrated = export_calibrated_run(
                target, dataset, root / "history", target.name + "-calibrated"
            )
            diagnose(calibrated)

    if protocol and protocol.get("baseline"):
        from dodge_native_game.variants.pixel_repr_ddqn.pixel_diagnostics import (
            evaluate_pixels,
        )
        from dodge_native_game.variants.pixel_repr_ddqn.pretrain import load_model
        from dodge_native_game.variants.pixel_repr_ddqn.probe import make_decoder

        baseline = root / "baseline"
        for name, digest in protocol["baseline"].items():
            assert file_hash(baseline / name) == digest
        model, payload = load_model(baseline / "checkpoint.pt")
        assert payload["data_hash"] == protocol["data_hash"]
        decoder_payload = torch.load(
            baseline / "decoder.pt", map_location="cpu", weights_only=True
        )
        assert (
            decoder_payload["world_model_sha256"]
            == protocol["baseline"]["checkpoint.pt"]
        )
        decoder = make_decoder(decoder_payload["latent_dim"])
        decoder.load_state_dict(decoder_payload["model"])
        checks = evaluate_pixels(
            model.to("cuda"),
            decoder.to("cuda").eval(),
            PixelSequenceDataset(dataset, split="train"),
            PixelSequenceDataset(dataset, split="validation"),
            "cuda",
        )
        checks.update(
            checkpoint_sha256=protocol["baseline"]["checkpoint.pt"],
            decoder_sha256=protocol["baseline"]["decoder.pt"],
            data_hash=protocol["data_hash"],
        )
        atomic_json(run / "baseline_pixel_diagnostics.json", checks)
        for name, digest in protocol["baseline"].items():
            assert file_hash(baseline / name) == digest

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
