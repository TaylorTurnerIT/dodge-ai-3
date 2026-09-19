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


def test_publish_results_copies_files_and_trace_tree(tmp_path: Path) -> None:
    launcher = _load("survival_launcher_publish", "colab_survival_study.py")
    extracted = tmp_path / "extracted"
    (extracted / "trace" / "mpc" / "scene-0").mkdir(parents=True)
    (extracted / "probe.json").write_text("{}")
    (extracted / "trace" / "mpc" / "scene-0" / "steps.jsonl").write_text(
        '{"step": 0}\n'
    )
    (extracted / "trace" / "mpc" / "scene-0" / "frame_000.png").write_bytes(
        b"\x89PNG\r\n\x1a\n"
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    launcher._publish_results(extracted, run_dir)
    assert (run_dir / "probe.json").is_file()
    assert not (run_dir / "mpc-report.json").exists()
    assert (run_dir / "trace" / "mpc" / "scene-0" / "steps.jsonl").is_file()
    assert (run_dir / "trace" / "mpc" / "scene-0" / "frame_000.png").is_file()


def test_survival_remote_driver_sets_up_before_fitting() -> None:
    launcher = _load("survival_launcher_fixture", "colab_survival_study.py")
    driver = launcher.build_remote_driver(
        source_hash="ab" * 32,
        run_id="survival-test",
        wheel=None,
        site_packages=["pytest"],
    )
    compile(driver, "<survival-remote-driver>", "exec")
    setup = driver.index("uv_ok=False")
    native = driver.index("dodge-python")
    worker = driver.index("colab_survival_worker.py")
    smoke = driver.index("'smoke'")
    scored = driver.index("'scored'")
    complete = driver.index("SURVIVAL_LAUNCHER_COMPLETE")
    assert setup < native < worker < smoke < scored < complete
    assert "pytest" in driver
    assert "UV_SETUP_FALLBACK" in driver
    assert "WHEEL_CAPTURED" in driver
    assert "SURVIVAL_DRIVER_COMPLETE" in driver
    assert "sys.executable,'-m','uv'" in driver


def test_survival_protocol_pins_frozen_inputs_and_gate(tmp_path: Path) -> None:
    launcher = _load("survival_launcher_fixture", "colab_survival_study.py")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fake-checkpoint")
    probe_set = tmp_path / "probe-set"
    probe_set.mkdir()
    (probe_set / "manifest.json").write_bytes(b'{"fixture":true}\n')
    protocol, bundle = launcher.build_protocol(
        run_id="survival-test",
        checkpoint=checkpoint,
        probe_set=probe_set,
        source_commit="c" * 40,
    )
    assert protocol["experiment"] == "lewm-survival-trackb-v1"
    assert set(protocol["inputs"]) == {
        "checkpoint.pt",
        "probe-set/manifest.json",
    }
    assert protocol["inputs"]["checkpoint.pt"] == launcher.digest(checkpoint)
    assert protocol["probe"] == {
        "steps": 512,
        "batch_size": 256,
        "seed": 905,
        "max_windows": 20000,
    }
    assert protocol["gate"]["min_val_auprc"] == 0.10
    assert protocol["mpc"]["policies"] == {
        "mpc_h3": "mpc_h3",
        "mpc_h3_last": "mpc_h3_last",
        "random": "random",
        "neutral": "neutral",
    }
    assert protocol["mpc"]["scenario_cohorts"] == [0, 1]
    assert protocol["mpc"]["history_size"] == 3
    assert protocol["mpc"]["max_decisions"] == 128
    assert bundle["checkpoint.pt"] == checkpoint


def test_survival_targets_20k_confirmation_round() -> None:
    launcher = _load("survival_launcher_fixture", "colab_survival_study.py")
    assert (
        launcher.CHECKPOINT_RELPATH == "scale-resume-20000/checkpoint.pt"
    )
    assert launcher.SCENARIO_COHORTS == (0, 1)


def test_survival_site_packages_pin_transformers() -> None:
    launcher = _load("survival_launcher_fixture", "colab_survival_study.py")
    assert "transformers==4.57.6" in launcher.SITE_PACKAGES
    assert launcher.EXPECTED_TRANSFORMERS == "4.57.6"
    driver = launcher.build_remote_driver(
        source_hash="ab" * 32,
        run_id="survival-test",
        wheel=None,
        site_packages=list(launcher.SITE_PACKAGES),
    )
    assert "transformers==4.57.6" in driver


def test_survival_protocol_rejects_missing_inputs(tmp_path: Path) -> None:
    launcher = _load("survival_launcher_fixture", "colab_survival_study.py")
    with pytest.raises(ValueError, match="frozen input is missing"):
        launcher.build_protocol(
            run_id="survival-test",
            checkpoint=tmp_path / "absent.pt",
            probe_set=tmp_path / "absent-set",
            source_commit="c" * 40,
        )


def test_smoke_batch_loads_real_windows(tmp_path: Path) -> None:
    import numpy as np

    from dodge_native_game.variants.pixel_repr_ddqn.survival_probe import (
        PROBE_FORMAT,
    )

    worker = _load("survival_worker_fixture", "colab_survival_worker.py")
    root = tmp_path / "probe-set"
    episode = root / "episodes" / "train" / "episode.npz"
    episode.parent.mkdir(parents=True)
    frames = np.zeros((9, 3, 128, 128), dtype=np.uint8)
    frames[:, 2] = 200
    actions = np.zeros(8, dtype=np.int64)
    terminated = np.zeros(8, dtype=np.bool_)
    truncated = np.zeros(8, dtype=np.bool_)
    truncated[-1] = True
    with episode.open("wb") as stream:
        np.savez_compressed(
            stream,
            pixels=frames,
            actions=actions,
            terminated=terminated,
            truncated=truncated,
        )
    manifest = {
        "format": PROBE_FORMAT,
        "splits": {"train": 1, "validation": 0},
        "deaths": {"train": 0, "validation": 0},
        "episodes": {
            "train": [
                {
                    "episode_id": "train-000000",
                    "recipe_family": "test",
                    "seed": 22000,
                    "path": "episodes/train/episode.npz",
                    "sha256": "x",
                    "decisions": 8,
                    "terminated": False,
                }
            ],
            "validation": [],
        },
    }
    (root / "manifest.json").write_text(
        __import__("json").dumps(manifest) + "\n"
    )
    batch = worker._smoke_batch(root)
    assert tuple(batch.shape) == (2, 4, 3, 128, 128)
    assert str(batch.dtype) == "torch.uint8"


def test_probe_gate_decision_fails_closed() -> None:
    worker = _load("survival_worker_fixture", "colab_survival_worker.py")
    assert worker.probe_gate_decision({"val_auprc": None}, 0.10) == {
        "passed": False,
        "reason": "val_auprc is null; no ranking signal",
    }
    failed = worker.probe_gate_decision({"val_auprc": 0.03}, 0.10)
    assert failed["passed"] is False
    passed = worker.probe_gate_decision({"val_auprc": 0.42}, 0.10)
    assert passed == {"passed": True, "reason": "val_auprc 0.4200 meets 0.1000"}
    import json

    json.dumps(passed, allow_nan=False)
