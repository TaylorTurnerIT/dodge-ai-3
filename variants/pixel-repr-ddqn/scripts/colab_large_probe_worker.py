"""Extract frozen representations, fit matched heads, and package T4 evidence."""


def _assert_metric_close(expected, actual, *, label: str) -> float | None:
    import math

    if expected is None or actual is None:
        if expected is not actual:
            raise RuntimeError(f"baseline {label} changed from null to a value")
        return None
    expected_value = float(expected)
    actual_value = float(actual)
    if not math.isfinite(expected_value) or not math.isfinite(actual_value):
        raise RuntimeError(f"baseline {label} is nonfinite")
    difference = abs(actual_value - expected_value)
    if difference > 1e-6:
        raise RuntimeError(
            f"baseline {label} changed by {difference:.9g}, expected <= 1e-6"
        )
    return difference


def _baseline_reevaluation(root, protocol, run_id, *, device):
    import json

    import torch

    from dodge_native_game.variants.pixel_repr_ddqn.large_probe import (
        evaluate_decoder_stream,
        open_frame_bank,
        stream_train_mean,
    )
    from dodge_native_game.variants.pixel_repr_ddqn.query_decoder import (
        QueryPixelDecoder,
    )
    from dodge_native_game.variants.pixel_repr_ddqn.run_artifacts import file_hash

    bank = open_frame_bank(
        root / "dataset",
        root / "probe-banks",
        checkpoint_sha256=protocol["checkpoint_sha256"],
    )
    training_mean = stream_train_mean(bank.pixels, bank.indices("train"))
    output = {
        "run_id": run_id,
        "loss_kind": protocol["loss_kind"],
        "baseline_loss_kind": "mse",
        "baseline_run_id": protocol["baseline_run_id"],
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "data_sha256": bank.data_hash,
        "frame_index_sha256": bank.frame_index_hash,
        "tolerance": 1e-6,
        "representations": {},
        "passed": False,
    }
    for mode in protocol["representations"]:
        files = protocol["baseline_files"][mode]
        decoder_path = root / "baselines" / mode / "decoder-8192.pt"
        evaluation_path = root / "baselines" / mode / "evaluation-8192.json"
        if file_hash(decoder_path) != files["decoder-8192.pt"]:
            raise RuntimeError(f"baseline {mode} decoder hash mismatch")
        if file_hash(evaluation_path) != files["evaluation-8192.json"]:
            raise RuntimeError(f"baseline {mode} evaluation hash mismatch")
        original = json.loads(evaluation_path.read_text())
        if (
            original.get("step") != 8192
            or original.get("representation") != mode
            or original.get("world_model_sha256") != protocol["checkpoint_sha256"]
            or original.get("data_sha256") != bank.data_hash
            or original.get("frame_index_sha256") != bank.frame_index_hash
        ):
            raise RuntimeError(f"baseline {mode} is not from this frozen bank")
        payload = torch.load(decoder_path, map_location="cpu", weights_only=True)
        if (
            payload.get("decoder_kind") != mode
            or payload.get("step") != 8192
            or payload.get("latent_dim") != 192
            or payload.get("loss_kind", "mse") != "mse"
            or payload.get("world_model_sha256") != protocol["checkpoint_sha256"]
            or payload.get("data_sha256") != bank.data_hash
            or payload.get("frame_index_sha256") != bank.frame_index_hash
        ):
            raise RuntimeError(f"baseline {mode} decoder provenance mismatch")
        decoder = QueryPixelDecoder(192).to(device).eval()
        decoder.load_state_dict(payload["model"], strict=True)
        with torch.no_grad():
            reevaluated = evaluate_decoder_stream(
                decoder,
                bank.features[mode],
                bank.pixels,
                bank.changed,
                bank.records,
                bank.split_ranges,
                training_mean,
                device=device,
                batch_size=64,
            )
        comparisons = {}
        for split in ("train", "validation"):
            original_metrics = original["splits"][split]
            metrics = reevaluated["splits"][split]
            comparisons[split] = {
                "reported": {
                    "mse": original_metrics.get("mse"),
                    "changed_region_mse": original_metrics.get(
                        "changed_region_mse"
                    ),
                },
                "global_mse_abs_delta": _assert_metric_close(
                    original_metrics.get("mse"),
                    metrics.get("mse"),
                    label=f"{mode} {split} mse",
                ),
                "changed_mse_abs_delta": _assert_metric_close(
                    original_metrics.get("changed_region_mse"),
                    metrics.get("changed_region_mse"),
                    label=f"{mode} {split} changed mse",
                ),
                "reevaluated": metrics,
            }
        output["representations"][mode] = {
            "decoder_sha256": files["decoder-8192.pt"],
            "evaluation_sha256": files["evaluation-8192.json"],
            "splits": comparisons,
        }
    output["passed"] = True
    destination = root / "history" / f"{run_id}-baseline-reevaluation.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.write_text(json.dumps(output, indent=2) + "\n")
    return output


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
        loss_kind=protocol["loss_kind"],
        device="cuda",
        bank_root=root / "probe-banks",
    )
    assert file_hash(root / "checkpoint.pt") == protocol["checkpoint_sha256"]
    if protocol["loss_kind"] == "balanced-bright" and protocol["baseline_files"]:
        _baseline_reevaluation(root, protocol, run_id, device="cuda")
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
