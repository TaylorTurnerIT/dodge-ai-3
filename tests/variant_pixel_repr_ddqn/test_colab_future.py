from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _worker_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "variants/pixel-repr-ddqn/scripts/colab_future_worker.py"
    )
    spec = importlib.util.spec_from_file_location("future_worker_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _protocol(readout: str = "predicted-latent-broadcast") -> dict:
    protocol = {
        "experiment": "lewm-future-decode-v1",
        "run_id": "future-remote",
        "checkpoint_name": "world.pt",
        "checkpoint_sha256": "a" * 64,
        "current_decoder_name": "current-decoder.pt",
        "current_decoder_sha256": "c" * 64,
        "data_sha256": "d" * 64,
        "milestones": [512, 2048],
        "readout": readout,
    }
    if readout == "actual-next-latent":
        protocol["ab_decoder_name"] = "ab-decoder.pt"
        protocol["ab_decoder_sha256"] = "b" * 64
        protocol["ab_source_run"] = "future-remote-ab"
    return protocol


def _result(readout: str = "predicted-latent-broadcast") -> dict:
    result = {
        "experiment": "lewm-future-decode-v1",
        "world_model_sha256": "a" * 64,
        "data_sha256": "d" * 64,
        "current_decoder_sha256": "c" * 64,
        "milestones": [512, 2048],
        "world_model_updates": 0,
        "batch_size": 32,
        "parameter_count": 165056,
        "scene_count": 16,
        "loss_kind": "balanced-bright",
        "equal_class_weights": True,
        "readout": readout,
    }
    if readout == "actual-next-latent":
        result["ab_decoder_sha256"] = "b" * 64
        result["ab_source_run"] = "future-remote-ab"
    return result


def test_future_worker_accepts_matching_artifact_contract() -> None:
    worker = _worker_module()
    worker.validate_result(_result(), _protocol())


def test_future_worker_accepts_actual_readout_contract() -> None:
    worker = _worker_module()
    worker.validate_result(
        _result("actual-next-latent"), _protocol("actual-next-latent")
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("world_model_updates", 1),
        ("loss_kind", "palette-ce"),
        ("current_decoder_sha256", "x" * 64),
        ("parameter_count", 165057),
        ("scene_count", 8),
        ("batch_size", 16),
        ("readout", "actual-next-latent"),
    ],
)
def test_future_worker_rejects_contract_drift(key: str, value) -> None:
    worker = _worker_module()
    result = _result()
    result[key] = value
    with pytest.raises(RuntimeError, match="contract mismatch"):
        worker.validate_result(result, _protocol())


def test_future_worker_rejects_actual_readout_drift() -> None:
    worker = _worker_module()
    result = _result("actual-next-latent")
    result["ab_source_run"] = "other-run"
    with pytest.raises(RuntimeError, match="contract mismatch"):
        worker.validate_result(result, _protocol("actual-next-latent"))


def test_future_worker_roots_default_and_overlay(monkeypatch) -> None:
    from pathlib import Path

    worker = _worker_module()
    monkeypatch.delenv("LEWM_WORK_ROOT", raising=False)
    monkeypatch.delenv("LEWM_CODE_ROOT", raising=False)
    work, code = worker._work_roots()
    assert work == Path("/content/lewm-work")
    assert code == Path("/content/lewm-work")
    monkeypatch.setenv("LEWM_WORK_ROOT", "/content/lewm-ac2-work")
    monkeypatch.setenv("LEWM_CODE_ROOT", "/content/lewm-ac2-src")
    work, code = worker._work_roots()
    assert work == Path("/content/lewm-ac2-work")
    assert code == Path("/content/lewm-ac2-src")
