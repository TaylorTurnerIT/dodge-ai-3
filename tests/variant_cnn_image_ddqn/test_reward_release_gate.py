"""Exercise release/failure handling without allocating a GPU."""

import json
import runpy
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


@pytest.mark.parametrize("added,release", [(0.01, True), (0.06, False)])
def test_reward_gate_releases_only_after_all_pairs_pass(
    tmp_path, monkeypatch, added, release
):
    script = Path(__file__).resolve().parents[2] / "scripts/ddqn-reward-campaign.py"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "--gate-worker" in command:
            name = command[command.index("--gate-worker") + 1]
            duration = 100 * (1 + added) if "routine" in name else 100
            (tmp_path / f"{name}.json").write_text(
                json.dumps(
                    {
                        "training_loop_seconds": duration,
                        "optimizer_steps": 2250,
                    }
                )
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script), "--history", str(tmp_path), "--treatments", "rgbcontrol"],
    )
    if release:
        runpy.run_path(str(script), run_name="__main__")
    else:
        with pytest.raises(RuntimeError, match="telemetry gate failed"):
            runpy.run_path(str(script), run_name="__main__")
    assert sum("--gate-worker" in call for call in calls) == 6
    launches = [
        call for call in calls if any("ddqn-t4-screen.py" in arg for arg in call)
    ]
    assert bool(launches) == release
    failure = tmp_path / "campaign-failure.tar.gz"
    assert failure.exists() != release
    if failure.exists():
        with tarfile.open(failure) as archive:
            assert "reward_telemetry_gate.json" in archive.getnames()
