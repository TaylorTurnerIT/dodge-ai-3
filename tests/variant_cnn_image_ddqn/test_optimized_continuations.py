import importlib.util
from pathlib import Path


def test_continuation_treatments_have_distinct_immutable_outputs():
    path = (
        Path(__file__).resolve().parents[2] / "scripts/ddqn-optimized-continuations.py"
    )
    spec = importlib.util.spec_from_file_location("continuations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    treatments = module.TREATMENTS
    assert len({row["run_id"] for row in treatments.values()}) == len(treatments)
    assert treatments["fresh-best"]["resume"] is None
    assert treatments["best-resume"]["steps"] == 300_000
    assert treatments["boundary10-resume"]["reward_profile"] == "boundary10-v1"
    assert all(
        "500k" in row["run_id"] for name, row in treatments.items() if "resume" in name
    )
    assert "smoke_revision" in path.read_text()
