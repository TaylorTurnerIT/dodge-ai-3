"""Target refresh sweep must not change the reference learner treatment."""

import importlib.util
from pathlib import Path


def test_sync_treatments_change_only_target_interval():
    script = Path(__file__).resolve().parents[2] / "scripts/ddqn-t4-screen.py"
    spec = importlib.util.spec_from_file_location("sync_screen", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = module.treatment("nstep3s43boundary")
    for interval in (100, 1000, 2500, 5000, 10000):
        candidate = module.treatment(f"boundarysync{interval}")
        assert candidate.pop("target_sync_interval") == interval
        assert candidate == reference
