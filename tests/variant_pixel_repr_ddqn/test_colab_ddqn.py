from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _scripts() -> Path:
    return (
        Path(__file__).resolve().parents[2] / "variants/pixel-repr-ddqn/scripts"
    )


def _load(name: str, filename: str):
    scripts = _scripts()
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    path = scripts / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _checkpoint(tmp_path: Path) -> Path:
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"fake-checkpoint")
    return ckpt


def test_ddqn_protocol_pins_training_hyperparams(tmp_path: Path) -> None:
    launcher = _load("ddqn_launcher_fixture", "colab_ddqn_study.py")
    protocol, bundle = launcher.build_protocol(
        run_id="ddqn-test",
        checkpoint=_checkpoint(tmp_path),
        source_commit="c" * 40,
    )
    assert protocol["experiment"] == "lewm-ddqn-screen-v1"
    assert protocol["inputs"]["checkpoint.pt"] == launcher.digest(
        tmp_path / "checkpoint.pt"
    )
    training = protocol["training"]
    assert training["seed"] == 7
    assert training["decisions"] == 200_000
    assert training["warmup_decisions"] == 5_000
    assert training["epsilon_start"] == 1.0
    assert training["epsilon_final"] == 0.05
    assert training["epsilon_decay_decisions"] == 120_000
    assert training["gamma"] == 0.99
    assert training["batch_size"] == 32
    assert training["sync_interval"] == 1000
    assert training["replay_capacity"] == 100_000
    assert training["scenario_cohorts"] == [0, 1]
    assert training["eval_every_decisions"] == 20_000
    assert bundle["checkpoint.pt"] == tmp_path / "checkpoint.pt"
    with pytest.raises(ValueError, match="missing"):
        launcher.build_protocol(
            run_id="ddqn-test",
            checkpoint=tmp_path / "absent.pt",
            source_commit="c" * 40,
        )


def test_ddqn_remote_driver_smokes_then_screens() -> None:
    launcher = _load("ddqn_launcher_driver", "colab_ddqn_study.py")
    wheel = {"filename": "dodge_native-0.1-py3-none-any.whl", "sha256": "ab" * 32}
    driver = launcher.build_remote_driver(
        source_hash="ab" * 32,
        run_id="ddqn-test",
        wheel=wheel,
        site_packages=["pytest"],
    )
    compile(driver, "<ddqn-remote-driver>", "exec")
    assert "lewm-ddqn.tar.gz" in driver
    assert "native_path = str(wheelhouse)" in driver
    assert "os.pathsep + native_path" in driver
    smoke = driver.index("'smoke'")
    scored = driver.index("'scored'")
    complete = driver.index("DDQN_LAUNCHER_COMPLETE")
    assert smoke < scored < complete
    assert "colab_ddqn_worker.py" in driver
    assert "transformers.__version__" in driver
    plain = launcher.build_remote_driver(
        source_hash="ab" * 32,
        run_id="ddqn-test",
        wheel=None,
        site_packages=["pytest"],
    )
    compile(plain, "<ddqn-remote-driver>", "exec")
    assert "native_path = ''" in plain


def test_ddqn_publish_requires_results_and_checkpoints(
    tmp_path: Path,
) -> None:
    launcher = _load("ddqn_launcher_publish", "colab_ddqn_study.py")
    extracted = tmp_path / "extracted"
    (extracted / "checkpoints").mkdir(parents=True)
    (extracted / "metrics.jsonl").write_text("{}\n")
    (extracted / "eval.json").write_text("[]")
    (extracted / "environment.json").write_text("{}")
    (extracted / "checkpoints" / "agent-final.pt").write_bytes(b"fake")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    launcher._publish_results(extracted, run_dir)
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "eval.json").is_file()
    assert (run_dir / "checkpoints" / "agent-final.pt").is_file()
    (extracted / "eval.json").unlink()
    run2 = tmp_path / "run2"
    run2.mkdir()
    with pytest.raises(RuntimeError, match="missing"):
        launcher._publish_results(extracted, run2)


def test_ddqn_epsilon_schedule_bounds() -> None:
    worker = _load("ddqn_worker_schedule", "colab_ddqn_worker.py")
    cfg = {
        "epsilon_start": 1.0,
        "epsilon_final": 0.05,
        "epsilon_decay_decisions": 120_000,
    }
    assert worker.epsilon_at(0, cfg) == 1.0
    assert worker.epsilon_at(60_000, cfg) == pytest.approx(0.525)
    assert worker.epsilon_at(120_000, cfg) == 0.05
    assert worker.epsilon_at(200_000, cfg) == 0.05
