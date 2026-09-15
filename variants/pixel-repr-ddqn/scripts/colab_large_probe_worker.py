"""Extract frozen representations, fit matched heads, and package T4 evidence."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path

PALETTE_LOSS_KINDS = ("palette-ce", "palette-bce")
PALETTE_SUFFIXES = {"palette-ce": "ce", "palette-bce": "bce"}


def _read_json(path: Path, label: str) -> dict[str, object]:
    import json

    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return value


def _palette_fields(
    payload: Mapping[str, object],
) -> tuple[list[object] | None, str | None]:
    """Read the exact palette provenance emitted by the probe core."""

    palette_rgb = payload.get("palette_rgb")
    if not isinstance(palette_rgb, list):
        palette_rgb = None
    palette_hash = payload.get("palette_sha256")
    if not isinstance(palette_hash, str) or not palette_hash:
        palette_hash = None
    return palette_rgb, palette_hash


def _validate_palette_provenance(
    payload: Mapping[str, object], label: str
) -> tuple[list[object], str]:
    palette_rgb, palette_hash = _palette_fields(payload)
    if palette_rgb is None or palette_hash is None:
        raise RuntimeError(f"{label} palette provenance is incomplete")
    if not 1 <= len(palette_rgb) <= 256:
        raise RuntimeError(f"{label} palette size is invalid")
    packed = bytearray()
    previous: tuple[int, int, int] | None = None
    for index, color in enumerate(palette_rgb):
        if not isinstance(color, list) or len(color) != 3:
            raise RuntimeError(f"{label} palette color {index} is invalid")
        if any(
            isinstance(channel, bool)
            or not isinstance(channel, int)
            or not 0 <= channel <= 255
            for channel in color
        ):
            raise RuntimeError(f"{label} palette color {index} is invalid")
        normalized = (color[0], color[1], color[2])
        if previous is not None and normalized <= previous:
            raise RuntimeError(f"{label} palette is not strictly sorted")
        packed.extend(normalized)
        previous = normalized
    if hashlib.sha256(bytes(packed)).hexdigest() != palette_hash:
        raise RuntimeError(f"{label} palette hash does not match RGB entries")
    return palette_rgb, palette_hash


def _has_train_only_input(payload: Mapping[str, object]) -> bool:
    return payload.get("palette_source_split") == "train"


def _loss_kinds(protocol: Mapping[str, object]) -> tuple[str, ...]:
    values = protocol.get("loss_kinds")
    if values is None:
        value = protocol.get("loss_kind")
        if not isinstance(value, str):
            raise RuntimeError("protocol does not declare loss_kind")
        return (value,)
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise RuntimeError("protocol loss_kinds must be a string list")
    result = tuple(values)
    if not result:
        raise RuntimeError("protocol loss_kinds cannot be empty")
    return result


def _condition_run_id(run_id: str, loss_kind: str, comparison: bool) -> str:
    if not comparison:
        return run_id
    try:
        suffix = PALETTE_SUFFIXES[loss_kind]
    except KeyError as error:
        raise RuntimeError(
            f"unsupported palette comparison loss {loss_kind}"
        ) from error
    return f"{run_id}-{suffix}"


def _palette_condition_evidence(
    root: Path,
    protocol: Mapping[str, object],
    run_id: str,
    loss_kind: str,
) -> dict[str, object]:
    """Validate one palette arm and return its comparable provenance."""

    condition_id = _condition_run_id(run_id, loss_kind, True)
    metadata: list[Mapping[str, object]] = []
    for mode in ("cls", "projected"):
        run = root / "history" / f"{condition_id}-{mode}"
        report = _read_json(run / "report.json", f"{run.name}/report.json")
        config = _read_json(run / "config.json", f"{run.name}/config.json")
        manifest = _read_json(run / "manifest.json", f"{run.name}/manifest.json")
        metadata.extend((report, config, manifest))
    comparison = _read_json(
        root / "history" / f"{condition_id}-comparison.json",
        f"{condition_id}-comparison.json",
    )
    metadata.append(comparison)
    palette_values = [
        _validate_palette_provenance(payload, condition_id) for payload in metadata
    ]
    palette_rgb, palette_hash = palette_values[0]
    if any(value != (palette_rgb, palette_hash) for value in palette_values):
        raise RuntimeError(f"{condition_id} palette provenance differs between heads")
    if any(not _has_train_only_input(payload) for payload in metadata):
        raise RuntimeError(f"{condition_id} does not declare train-only input")
    expected_world = protocol["checkpoint_sha256"]
    expected_data = protocol["data_hash"]
    if comparison.get("loss_kind") != loss_kind:
        raise RuntimeError(f"{condition_id} loss provenance mismatch")
    if comparison.get("world_model_sha256") != expected_world:
        raise RuntimeError(f"{condition_id} checkpoint provenance mismatch")
    if comparison.get("data_sha256") != expected_data:
        raise RuntimeError(f"{condition_id} dataset provenance mismatch")
    if comparison.get("palette_source_split") != "train":
        raise RuntimeError(f"{condition_id} decoder input is not train-only")
    return {
        "run_id": condition_id,
        "loss_kind": loss_kind,
        "palette_rgb": palette_rgb,
        "palette_sha256": palette_hash,
        "palette_source_split": "train",
        "world_model_sha256": expected_world,
        "data_sha256": expected_data,
        "frame_index_sha256": comparison.get("frame_index_sha256"),
        "milestones": comparison.get("milestones"),
        "representations": ["cls", "projected"],
    }


def _write_palette_comparison(
    root: Path,
    protocol: Mapping[str, object],
    run_id: str,
    evidence: Mapping[str, Mapping[str, object]],
    source_hash: str,
) -> dict[str, object]:
    import json

    identities = [
        (value.get("palette_rgb"), value.get("palette_sha256"))
        for value in evidence.values()
    ]
    if not identities or any(value != identities[0] for value in identities[1:]):
        raise RuntimeError("CE/BCE palette identity/hash differs")
    palette_rgb, palette_hash = identities[0]
    if not isinstance(palette_rgb, list) or not isinstance(palette_hash, str):
        raise RuntimeError("CE/BCE palette identity/hash is incomplete")
    _validate_palette_provenance(
        {"palette_rgb": palette_rgb, "palette_sha256": palette_hash},
        "palette comparison",
    )
    data_hashes = {value.get("data_sha256") for value in evidence.values()}
    frame_hashes = {value.get("frame_index_sha256") for value in evidence.values()}
    world_hashes = {value.get("world_model_sha256") for value in evidence.values()}
    if data_hashes != {protocol["data_hash"]} or world_hashes != {
        protocol["checkpoint_sha256"]
    }:
        raise RuntimeError("CE/BCE frozen source provenance differs")
    frame_hash = next(iter(frame_hashes), None)
    if (
        len(frame_hashes) != 1
        or not isinstance(frame_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", frame_hash)
    ):
        raise RuntimeError("CE/BCE frame-index provenance differs")
    output = {
        "variant": "pixel-repr-ddqn",
        "comparison": "palette-loss",
        "run_id": run_id,
        "loss_kinds": list(PALETTE_LOSS_KINDS),
        "conditions": dict(evidence),
        "palette_rgb": palette_rgb,
        "palette_sha256": palette_hash,
        "palette_source_split": "train",
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "data_sha256": protocol["data_hash"],
        "frame_index_sha256": frame_hash,
        "milestones": protocol["milestones"],
        "world_model_fits": False,
        "source_sha256": source_hash,
    }
    destination = root / "history" / f"{run_id}-palette-comparison.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    return output


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
    loss_kinds = _loss_kinds(protocol)
    palette_comparison = protocol.get("palette_comparison") is True
    if palette_comparison:
        if loss_kinds != PALETTE_LOSS_KINDS:
            raise RuntimeError("palette comparison must run CE then BCE")
        for loss_kind in loss_kinds:
            run_study(
                root / "checkpoint.pt",
                root / "dataset",
                root / "history",
                _condition_run_id(run_id, loss_kind, True),
                milestones=tuple(protocol["milestones"]),
                batch_size=protocol["batch_size"],
                loss_kind=loss_kind,
                device="cuda",
                bank_root=root / "probe-banks",
            )
            assert file_hash(root / "checkpoint.pt") == protocol["checkpoint_sha256"]
        evidence = {
            loss_kind: _palette_condition_evidence(
                root, protocol, run_id, loss_kind
            )
            for loss_kind in loss_kinds
        }
        _write_palette_comparison(
            root,
            protocol,
            run_id,
            evidence,
            os.environ["LEWM_SOURCE_HASH"],
        )
    else:
        if len(loss_kinds) != 1 or protocol.get("loss_kind") != loss_kinds[0]:
            raise RuntimeError("single-condition protocol loss mismatch")
        run_study(
            root / "checkpoint.pt",
            root / "dataset",
            root / "history",
            run_id,
            milestones=tuple(protocol["milestones"]),
            batch_size=protocol["batch_size"],
            loss_kind=loss_kinds[0],
            device="cuda",
            bank_root=root / "probe-banks",
        )
        assert file_hash(root / "checkpoint.pt") == protocol["checkpoint_sha256"]
    if not palette_comparison and loss_kinds[0] == "balanced-bright" and protocol[
        "baseline_files"
    ]:
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
