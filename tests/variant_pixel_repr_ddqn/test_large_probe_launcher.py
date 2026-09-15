from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _script(name: str):
    path = Path(__file__).parents[2] / "variants/pixel-repr-ddqn/scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_metric_comparison_preserves_null_and_enforces_tolerance() -> None:
    worker = _script("colab_large_probe_worker.py")
    assert worker._assert_metric_close(None, None, label="empty") is None
    assert worker._assert_metric_close(0.25, 0.25, label="equal") == 0
    assert worker._assert_metric_close(0.25, 0.2500005, label="rounding") < 1e-6
    for expected, actual in ((None, 0), (0, None), (0.25, 0.250002)):
        with pytest.raises(RuntimeError):
            worker._assert_metric_close(expected, actual, label="changed")


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_baseline_metric_comparison_rejects_nonfinite_values(invalid: float) -> None:
    worker = _script("colab_large_probe_worker.py")
    for expected, actual in ((invalid, 0.25), (0.25, invalid)):
        with pytest.raises(RuntimeError):
            worker._assert_metric_close(expected, actual, label="nonfinite")


def test_baseline_bundle_records_exact_files_and_requires_final_validation(
    tmp_path: Path,
) -> None:
    launcher = _script("colab_large_probe.py")
    for mode in ("cls", "projected"):
        run = tmp_path / f"baseline-{mode}"
        run.mkdir()
        (run / "decoder-8192.pt").write_bytes(mode.encode())
        (run / "evaluation-8192.json").write_text(
            json.dumps(
                {
                    "step": 8192,
                    "representation": mode,
                    "splits": {"validation": {"frame_count": 2048}},
                }
            )
        )
    bundle = launcher._baseline_bundle(tmp_path, "baseline")
    for mode in ("cls", "projected"):
        assert set(bundle[mode]) == {
            "decoder",
            "evaluation",
            "decoder-8192.pt",
            "evaluation-8192.json",
        }
        assert (
            bundle[mode]["decoder-8192.pt"] == hashlib.sha256(mode.encode()).hexdigest()
        )
    path = tmp_path / "baseline-projected/evaluation-8192.json"
    original = json.loads(path.read_text())
    original["splits"]["validation"]["frame_count"] = 2047
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="incomplete validation"):
        launcher._baseline_bundle(tmp_path, "baseline")


def test_palette_comparison_protocol_and_prefixes() -> None:
    launcher = _script("colab_large_probe.py")

    assert launcher._loss_kinds("mse", True) == ("palette-ce", "palette-bce")
    assert launcher._run_names(
        "screen",
        loss_kinds=("palette-ce", "palette-bce"),
        palette_comparison=True,
    ) == (
        "screen-ce-cls",
        "screen-ce-projected",
        "screen-bce-cls",
        "screen-bce-projected",
    )
    protocol = launcher._protocol(
        checkpoint_sha256="c" * 64,
        data_hash="d" * 64,
        run_id="screen",
        loss_kind="mse",
        palette_comparison=True,
        baseline_run_id=None,
        baseline_files={},
    )
    assert protocol["loss_kind"] is None
    assert protocol["loss_kinds"] == ["palette-ce", "palette-bce"]
    assert protocol["decoder_input_split"] == "train"
    assert protocol["train_only_input"] is True
    with pytest.raises(ValueError, match="cannot be combined"):
        launcher._loss_kinds("balanced-bright", True)


def test_palette_comparison_requires_exact_palette_and_train_source_metadata(
    tmp_path: Path,
) -> None:
    launcher = _script("colab_large_probe.py")
    history = tmp_path / "history"
    history.mkdir()
    palette = [[0, 0, 0], [255, 255, 255]]
    palette_hash = hashlib.sha256(bytes(sum(palette, []))).hexdigest()
    conditions = {}
    for loss_kind, suffix in (("palette-ce", "ce"), ("palette-bce", "bce")):
        conditions[loss_kind] = {
            "run_id": f"screen-{suffix}",
            "loss_kind": loss_kind,
            "palette_rgb": palette,
            "palette_sha256": palette_hash,
            "palette_source_split": "train",
        }
    payload = {
        "loss_kinds": ["palette-ce", "palette-bce"],
        "palette_rgb": palette,
        "palette_sha256": palette_hash,
        "palette_source_split": "train",
        "checkpoint_sha256": "c" * 64,
        "data_sha256": "d" * 64,
        "conditions": conditions,
    }
    path = history / "screen-palette-comparison.json"
    path.write_text(json.dumps(payload))
    protocol = {"checkpoint_sha256": "c" * 64, "data_hash": "d" * 64}
    assert launcher._validate_palette_comparison(history, "screen", protocol) == payload

    payload["conditions"]["palette-bce"]["palette_sha256"] = "b" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="hash"):
        launcher._validate_palette_comparison(history, "screen", protocol)


def test_worker_palette_comparison_record_keeps_identity_and_frozen_hashes(
    tmp_path: Path,
) -> None:
    worker = _script("colab_large_probe_worker.py")
    history = tmp_path / "history"
    history.mkdir()
    palette_hash = hashlib.sha256(
        bytes([0, 0, 0, 255, 255, 255])
    ).hexdigest()
    evidence = {
        loss_kind: {
            "run_id": f"screen-{suffix}",
            "loss_kind": loss_kind,
            "palette_rgb": [[0, 0, 0], [255, 255, 255]],
            "palette_sha256": palette_hash,
            "palette_source_split": "train",
            "world_model_sha256": "c" * 64,
            "data_sha256": "d" * 64,
            "frame_index_sha256": "f" * 64,
            "milestones": [512, 2048, 8192],
        }
        for loss_kind, suffix in (("palette-ce", "ce"), ("palette-bce", "bce"))
    }
    protocol = {
        "checkpoint_sha256": "c" * 64,
        "data_hash": "d" * 64,
        "milestones": [512, 2048, 8192],
    }
    output = worker._write_palette_comparison(
        tmp_path,
        protocol,
        "screen",
        evidence,
        "s" * 64,
    )
    assert output["palette_source_split"] == "train"
    assert output["palette_sha256"] == palette_hash
    assert output["conditions"] == evidence
    assert (
        json.loads((history / "screen-palette-comparison.json").read_text())
        == output
    )


def test_worker_palette_condition_requires_provenance_on_each_head(
    tmp_path: Path,
) -> None:
    worker = _script("colab_large_probe_worker.py")
    history = tmp_path / "history"
    palette = [[0, 0, 0], [255, 255, 255]]
    palette_hash = hashlib.sha256(bytes(sum(palette, []))).hexdigest()
    common = {
        "palette_rgb": palette,
        "palette_sha256": palette_hash,
        "palette_source_split": "train",
    }
    for mode in ("cls", "projected"):
        run = history / f"screen-ce-{mode}"
        run.mkdir(parents=True)
        for filename in ("manifest.json", "config.json", "report.json"):
            payload = {
                **common,
                "loss_kind": "palette-ce",
                "world_model_sha256": "c" * 64,
                "data_hash": "d" * 64,
            }
            if filename == "report.json":
                payload["data_sha256"] = "d" * 64
            (run / filename).write_text(json.dumps(payload))
    comparison = {
        **common,
        "loss_kind": "palette-ce",
        "world_model_sha256": "c" * 64,
        "data_sha256": "d" * 64,
        "frame_index_sha256": "f" * 64,
        "milestones": [512, 2048, 8192],
    }
    (history / "screen-ce-comparison.json").write_text(json.dumps(comparison))
    protocol = {"checkpoint_sha256": "c" * 64, "data_hash": "d" * 64}
    evidence = worker._palette_condition_evidence(
        tmp_path, protocol, "screen", "palette-ce"
    )
    assert evidence["palette_sha256"] == palette_hash

    broken = json.loads((history / "screen-ce-projected/config.json").read_text())
    broken.pop("palette_sha256")
    (history / "screen-ce-projected/config.json").write_text(json.dumps(broken))
    with pytest.raises(RuntimeError, match="incomplete"):
        worker._palette_condition_evidence(tmp_path, protocol, "screen", "palette-ce")
