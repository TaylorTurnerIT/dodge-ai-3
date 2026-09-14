"""Remote subprocess errors remain errors across notebook output streaming."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def runner():
    path = (
        Path(__file__).parents[2] / "variants/pixel-repr-ddqn/scripts/colab_remote.py"
    )
    spec = importlib.util.spec_from_file_location("lewm_colab_remote", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_command


def test_v32_worker_failure_is_not_silent(runner, capsys):
    with pytest.raises(RuntimeError, match="Remote command failed"):
        runner([sys.executable, "-c", "print('before-failure'); raise SystemExit(3)"])
    assert "before-failure" in capsys.readouterr().out


def test_v32_worker_output_is_forwarded(runner, capsys):
    runner([sys.executable, "-c", "print('worker-output')"])
    assert "worker-output" in capsys.readouterr().out
