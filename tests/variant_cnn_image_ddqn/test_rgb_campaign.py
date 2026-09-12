from __future__ import annotations

import importlib.util
from pathlib import Path


def _campaign():
    path = Path(__file__).resolve().parents[2] / "scripts/ddqn-rgb-campaign.py"
    spec = importlib.util.spec_from_file_location("rgb_campaign", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_target_treatments_differ_only_in_sync_interval():
    module = _campaign()
    first, second = module.parameters(1000), module.parameters(10000)
    assert first.pop("target_sync_interval") == 1000
    assert second.pop("target_sync_interval") == 10000
    assert first == second
    assert first["training_seed_count"] == 700
    assert first["observation_profile"] == "native-rgb-v1"


def test_telemetry_gate_requires_three_pairs_and_rejects_slow_trial():
    module = _campaign()
    fast = {"routine": {"training_loop_seconds": 104},
            "minimal": {"training_loop_seconds": 100}}
    slow = {"routine": {"training_loop_seconds": 106},
            "minimal": {"training_loop_seconds": 100}}
    assert module.gate_summary([fast] * 3)["passed"]
    assert not module.gate_summary([fast] * 2)["passed"]
    assert not module.gate_summary([fast, fast, slow])["passed"]
