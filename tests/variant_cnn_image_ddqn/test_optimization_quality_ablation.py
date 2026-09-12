from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[2] / "scripts/ddqn-optimization-quality-ablation.py"
    spec = importlib.util.spec_from_file_location("quality_ablation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quality_ablation_completes_factorial_against_existing_corner() -> None:
    treatments = _module().TREATMENTS
    assert {
        (
            value["collector_lanes"],
            value["collector_execution"],
            value["learner_backend"],
        )
        for value in treatments.values()
    } == {
        (8, "parallel", "baseline"),
        (1, "serial", "cuda-optimized"),
        (1, "serial", "baseline"),
    }
    assert len({value["run_id"] for value in treatments.values()}) == 3
