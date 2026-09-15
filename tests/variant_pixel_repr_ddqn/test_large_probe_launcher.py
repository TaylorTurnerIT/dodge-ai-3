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
