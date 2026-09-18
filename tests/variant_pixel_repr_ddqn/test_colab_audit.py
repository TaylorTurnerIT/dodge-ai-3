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


def test_audit_protocol_pins_checkpoints_and_dataset(tmp_path: Path) -> None:
    launcher = _load("audit_launcher_fixture", "colab_audit_study.py")
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"fake-checkpoint")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_bytes(b'{"fixture":true}\n')
    protocol, bundle = launcher.build_protocol(
        run_id="audit-test",
        checkpoints={"14336": ckpt},
        dataset=dataset,
        source_commit="c" * 40,
    )
    assert protocol["experiment"] == "lewm-dynamics-audit-v1"
    assert protocol["checkpoints"] == {"14336": launcher.digest(ckpt)}
    assert protocol["data_hash"] == launcher.digest(dataset / "manifest.json")
    assert protocol["splits"] == ["train", "validation"]
    assert protocol["fractions"] == [0.25, 0.5, 0.75]
    assert bundle["checkpoints/14336.pt"] == ckpt
    assert bundle["dataset/manifest.json"] == dataset / "manifest.json"
    with pytest.raises(ValueError, match="missing"):
        launcher.build_protocol(
            run_id="audit-test",
            checkpoints={"14336": tmp_path / "absent.pt"},
            dataset=dataset,
            source_commit="c" * 40,
        )
    with pytest.raises(ValueError, match="at least one"):
        launcher.build_protocol(
            run_id="audit-test",
            checkpoints={},
            dataset=dataset,
            source_commit="c" * 40,
        )


def test_audit_remote_driver_runs_each_ckpt_and_split() -> None:
    launcher = _load("audit_launcher_driver", "colab_audit_study.py")
    driver = launcher.build_remote_driver(
        source_hash="ab" * 32,
        run_id="audit-test",
        wheel=None,
        site_packages=["pytest"],
        checkpoints=["9216", "14336"],
    )
    compile(driver, "<audit-remote-driver>", "exec")
    setup = driver.index("uv_ok = False")
    native = driver.index("dodge-python")
    runner = driver.index("run_dynamics_audit.py")
    loop = driver.index("for step in CKPTS:")
    complete = driver.index("AUDIT_LAUNCHER_COMPLETE")
    assert setup < native < runner < loop < complete
    assert "pytest" in driver
    assert "UV_SETUP_FALLBACK" in driver
    assert "transformers.__version__" in driver
    assert "9216" in driver and "14336" in driver
    assert "--device" in driver and "cuda" in driver


def test_audit_publish_requires_every_ckpt_file(tmp_path: Path) -> None:
    launcher = _load("audit_launcher_publish", "colab_audit_study.py")
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    (extracted / "audit-ckpt-14336-train.json").write_text("{}")
    (extracted / "audit-ckpt-14336-validation.json").write_text("{}")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    launcher._publish_results(extracted, run_dir, ["14336"])
    assert (run_dir / "audit-ckpt-14336-train.json").is_file()
    assert (run_dir / "audit-ckpt-14336-validation.json").is_file()
    with pytest.raises(RuntimeError, match="missing"):
        launcher._publish_results(extracted, tmp_path / "run2", ["9216"])
