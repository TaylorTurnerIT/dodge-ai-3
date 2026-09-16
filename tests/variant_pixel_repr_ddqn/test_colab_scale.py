from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _launcher(monkeypatch=None):
    scripts = (
        Path(__file__).resolve().parents[2]
        / "variants/pixel-repr-ddqn/scripts"
    )
    if monkeypatch is not None:
        monkeypatch.syspath_prepend(str(scripts))
    elif str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    path = scripts / "colab_scale_study.py"
    spec = importlib.util.spec_from_file_location("scale_launcher_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scale_remote_driver_sets_up_before_fitting() -> None:
    launcher = _launcher()
    driver = launcher.build_remote_driver(
        source_hash="ab" * 32,
        run_id="scale-test",
        site_packages=["pytest"],
        wheel_name=None,
    )
    compile(driver, "<scale-remote-driver>", "exec")
    setup = driver.index("uv_ok=False")
    native = driver.index("dodge-python")
    worker = driver.index("colab_scale_worker.py")
    complete = driver.index("SCALE_DRIVER_COMPLETE")
    assert setup < native < worker < complete
    assert "pytest" in driver
    assert "UV_SETUP_FALLBACK" in driver
    assert "WHEEL_CAPTURED" in driver
    assert "SCALE_RETURNCODE" in driver
    assert "uv_binary" not in driver


def test_scale_protocol_pins_frozen_inputs(tmp_path: Path) -> None:
    launcher = _launcher()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_bytes(b"{\"fixture\":true}\n")
    base = tmp_path / "base"
    base.mkdir()
    (base / "checkpoint.pt").write_bytes(b"fake-checkpoint")
    (base / "sample-trace.json").write_bytes(b"[]")
    (base / "stochastic-trace.json").write_bytes(b"[]")
    protocol = launcher.build_scale_protocol(
        run_id="scale-test",
        dataset=dataset,
        base_checkpoint=base / "checkpoint.pt",
        base_step=1024,
        extra_steps=4096,
        checkpoint_every=1024,
        source_commit="c" * 40,
    )
    assert protocol["experiment"] == "lewm-scale-continuation-v1"
    assert protocol["input_arm"] == "palette"
    assert protocol["world_batch_size"] == 32
    assert set(protocol["inputs"]) == {
        "dataset/manifest.json",
        "base/checkpoint.pt",
        "base/sample-trace.json",
        "base/stochastic-trace.json",
    }
    assert (
        protocol["base_checkpoint_sha256"]
        == protocol["inputs"]["base/checkpoint.pt"]
    )


def test_scale_remote_driver_creates_work_before_renames() -> None:
    import re

    launcher = _launcher()
    driver = launcher.build_remote_driver(
        source_hash="ab" * 32,
        run_id="scale-test",
        site_packages=["pytest"],
        wheel_name=None,
    )
    compile(driver, "<scale-remote-driver>", "exec")
    mkdir = driver.index("lewm-scale-work');work.mkdir(")
    first_rename = driver.index(".rename(code)")
    assert mkdir < first_rename
    assert len(re.findall(r"rename\(code\)", driver)) == 1


def test_newest_mirrored_step_requires_complete_triple(tmp_path: Path) -> None:
    launcher = _launcher()
    job = tmp_path / "job"
    assert launcher._newest_mirrored_step(job) == 0
    mirror = job / "mirror-checkpoints"
    mirror.mkdir(parents=True)
    (mirror / "checkpoint-2048.pt").write_bytes(b"ckpt")
    assert launcher._newest_mirrored_step(job) == 0
    (mirror / "sample-trace-2048.json").write_bytes(b"[]")
    (mirror / "stochastic-trace-2048.json").write_bytes(b"[]")
    assert launcher._newest_mirrored_step(job) == 2048


def test_scale_remote_driver_wheel_hit_skips_toolchain() -> None:
    launcher = _launcher()
    driver = launcher.build_remote_driver(
        source_hash="ab" * 32,
        run_id="scale-test",
        site_packages=["pytest"],
        wheel_name="dodge_native-0.1.0-cp311-abi3-linux_x86_64.whl",
    )
    compile(driver, "<scale-remote-driver>", "exec")
    assert "WHEEL_CACHE_HIT" in driver
    assert "cargo" not in driver
    assert "WHEEL_CAPTURED" not in driver
    assert "code/'wheel'" in driver


def test_scale_remote_driver_without_wheel_builds_toolchain() -> None:
    launcher = _launcher()
    driver = launcher.build_remote_driver(
        source_hash="ab" * 32,
        run_id="scale-test",
        site_packages=["pytest"],
        wheel_name=None,
    )
    compile(driver, "<scale-remote-driver>", "exec")
    assert "WHEEL_CAPTURED" in driver
    assert "cargo" in driver


def test_scale_protocol_records_batch_and_precision(tmp_path: Path) -> None:
    launcher = _launcher()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_bytes(b"{\"fixture\":true}\n")
    base = tmp_path / "base"
    base.mkdir()
    (base / "checkpoint.pt").write_bytes(b"fake-checkpoint")
    (base / "sample-trace.json").write_bytes(b"[]")
    (base / "stochastic-trace.json").write_bytes(b"[]")
    protocol = launcher.build_scale_protocol(
        run_id="scale-test",
        dataset=dataset,
        base_checkpoint=base / "checkpoint.pt",
        base_step=5120,
        extra_steps=4096,
        checkpoint_every=1024,
        source_commit="c" * 40,
        batch_size=128,
        precision="bf16",
    )
    assert protocol["world_batch_size"] == 128
    assert protocol["precision"] == "bf16"
