from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_script(name: str, module_name: str):
    path = (
        Path(__file__).resolve().parents[2]
        / "variants/pixel-repr-ddqn/scripts"
        / name
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _worker():
    return _load_script("colab_pooling_worker.py", "pooling_worker_fixture")


def _protocol() -> dict:
    return {
        "experiment": "lewm-pooling-readout-v1",
        "run_id": "pooling-remote",
        "source_spatial_run": "spatial-remote",
        "milestones": [512, 2048],
        "world_model_updates": 0,
        "inputs": {
            "world.pt": "a" * 64,
            "dataset/manifest.json": "d" * 64,
        },
        "recovery_targets": {},
        "core_initial_state_sha256": "c" * 64,
        "frame_index_sha256": "f" * 64,
        "palette_sha256": "e" * 64,
        "arms": ["mean", "max", "attention", "grid4"],
    }


def _result(protocol: dict) -> dict:
    return {
        "world_model_sha256": protocol["inputs"]["world.pt"],
        "data_sha256": protocol["inputs"]["dataset/manifest.json"],
        "frame_index_sha256": protocol["frame_index_sha256"],
        "palette_sha256": protocol["palette_sha256"],
        "core_initial_state_sha256": protocol["core_initial_state_sha256"],
        "milestones": protocol["milestones"],
        "arms": protocol["arms"],
        "world_model_updates": 0,
        "core_parameter_count": 165056,
        "attention_extra_parameter_count": 192,
    }


def test_pooling_worker_accepts_matching_artifact_contract() -> None:
    worker = _worker()
    protocol = _protocol()
    worker.validate_result(_result(protocol), protocol)


@pytest.mark.parametrize(
    "key,value",
    [
        ("world_model_updates", 1),
        ("core_parameter_count", 165057),
        ("attention_extra_parameter_count", 0),
        ("milestones", [512]),
        ("arms", ["mean"]),
    ],
)
def test_pooling_worker_rejects_contract_drift(key: str, value) -> None:
    worker = _worker()
    protocol = _protocol()
    result = _result(protocol)
    result[key] = value
    with pytest.raises(RuntimeError, match="contract mismatch"):
        worker.validate_result(result, protocol)


def _launcher(monkeypatch):
    monkeypatch.syspath_prepend(
        str(
            Path(__file__).resolve().parents[2]
            / "variants/pixel-repr-ddqn/scripts"
        )
    )
    import colab_pooling_study as launcher

    return launcher


def _driver_kwargs(**overrides):
    params = {
        "source_hash": "ab" * 32,
        "run_id": "pooling-test",
        "wheel": None,
        "inputs_url": None,
        "inputs_archive_sha256": None,
        "site_packages": ["pytest"],
    }
    params.update(overrides)
    return params


def test_pooling_remote_driver_sets_up_before_scoring(monkeypatch) -> None:
    launcher = _launcher(monkeypatch)
    driver = launcher.build_remote_driver(**_driver_kwargs())
    compile(driver, "<remote-driver>", "exec")
    setup = driver.index("uv_binary")
    native = driver.index("dodge-python")
    smoke = driver.index("run_phase('smoke','smoke.log'")
    scored = driver.index("run_phase('scored','scored.log'")
    assert setup < native < smoke < scored
    assert "pytest" in driver
    assert "UV_SETUP_FALLBACK" in driver
    assert "LOG_TAIL" in driver
    assert "WHEEL_CAPTURED" in driver
    assert "POOLING_DRIVER_COMPLETE" in driver


def test_pooling_remote_driver_wheel_hit_skips_toolchain(monkeypatch) -> None:
    launcher = _launcher(monkeypatch)
    wheel = {"key": "k" * 40, "filename": "dodge_native-0.1.0.whl", "sha256": "c" * 64}
    driver = launcher.build_remote_driver(**_driver_kwargs(wheel=wheel))
    compile(driver, "<remote-driver>", "exec")
    assert "WHEEL_CACHE_HIT" in driver
    assert "cargo" not in driver
    assert "WHEEL_CAPTURED" not in driver
    assert "code/'wheel'" in driver


def test_pooling_remote_driver_inputs_url_fetch(monkeypatch) -> None:
    launcher = _launcher(monkeypatch)
    driver = launcher.build_remote_driver(
        **_driver_kwargs(
            inputs_url="https://example.invalid/inputs.tar.gz",
            inputs_archive_sha256="d" * 64,
        )
    )
    compile(driver, "<remote-driver>", "exec")
    assert "LEWM_INPUTS_URL" in driver
    assert "INPUTS_ARCHIVE_VERIFIED" in driver
    fetch = driver.index("INPUTS_FETCH_START")
    smoke = driver.index("run_phase('smoke','smoke.log'")
    assert fetch < smoke


def test_pooling_split_archive_roundtrip(tmp_path: Path, monkeypatch) -> None:
    _launcher(monkeypatch)
    import colab_large_probe as probe

    archive = tmp_path / "archive.bin"
    archive.write_bytes(b"x" * (3 * 1024**2 + 7))
    parts = probe.split_archive(archive, tmp_path, part_size=1024**2)
    assert len(parts) == 4
    assert b"".join(p.read_bytes() for p in parts) == archive.read_bytes()
    with pytest.raises(ValueError, match="at least 1 MiB"):
        probe.split_archive(archive, tmp_path, part_size=512)


def test_pooling_upload_fans_out_and_assembles(tmp_path: Path, monkeypatch) -> None:
    _launcher(monkeypatch)
    import colab_large_probe as probe

    calls: list[tuple] = []

    def fake_cli(*args, **kwargs):
        calls.append(args)
        from types import SimpleNamespace

        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(probe, "cli", fake_cli)
    archive = tmp_path / "archive.bin"
    archive.write_bytes(b"y" * (2 * 1024**2 + 1))
    probe.upload(
        archive,
        "test-session",
        tmp_path,
        workers=2,
        part_size=1024**2,
        remote_prefix="lewm-pooling",
    )
    uploads = [c for c in calls if c[0] == "upload"]
    execs = [c for c in calls if c[0] == "exec"]
    assert len(uploads) == 3
    assert len(execs) == 1
    assert not list(tmp_path.glob("upload-*"))
    assembly = (tmp_path / "assemble.py").read_text()
    assert "lewm-pooling.part" in assembly
    assert "range(3)" in assembly


def test_pooling_worker_split_helpers() -> None:
    worker = _worker()
    targets = {
        "spatial-banks/standard/train/pixels.npy": "0" * 64,
        "spatial-banks/standard/validation/pixels.npy": "1" * 64,
        "spatial-banks/spatial/train/patches.npy": "2" * 64,
    }
    assert worker._standard_splits_needed(targets) == ["train", "validation"]
    assert worker._spatial_splits_needed(targets) == ["train"]


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _provenance_tree(root: Path) -> None:
    for kind in ("standard", "spatial"):
        for split in ("train", "validation"):
            (root / kind / split).mkdir(parents=True, exist_ok=True)
            (root / kind / split / "READY").write_text("ready\n")
            if kind == "standard":
                metadata = {"files": {}, "index_sha256": None}
                index_digest = _write(
                    root / kind / split / "index.json",
                    b'{"split": "%s"}' % split.encode(),
                )
                metadata["index_sha256"] = index_digest
                (root / kind / split / "metadata.json").write_text(
                    json.dumps(metadata)
                )
            else:
                array_digest = _write(
                    root / kind / split / "patches.npy",
                    f"patches-{split}".encode(),
                )
                (root / kind / split / "metadata.json").write_text(
                    json.dumps({"patches_sha256": array_digest})
                )


def test_pooling_launcher_protocol_marks_only_missing_for_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(
        str(
            Path(__file__).resolve().parents[2]
            / "variants/pixel-repr-ddqn/scripts"
        )
    )
    import colab_pooling_study as launcher

    provenance = tmp_path / "provenance"
    _provenance_tree(provenance)
    world_digest = _write(provenance / "world.pt", b"world")
    dataset = tmp_path / "dataset"
    manifest_digest = _write(dataset / "manifest.json", b"manifest")
    comparison = {
        "world_model_sha256": world_digest,
        "data_sha256": manifest_digest,
        "initial_state_sha256": "c" * 64,
        "frame_index_sha256": "f" * 64,
        "palette_sha256": "e" * 64,
    }
    protocol, bundle = launcher.build_protocol(
        provenance=provenance,
        dataset=dataset,
        run_id="pooling-test",
        spatial_run="spatial-test",
        source_commit="deadbeef",
        comparison=comparison,
    )
    assert protocol["recovery_targets"] == {}
    assert protocol["inputs"]["world.pt"] == world_digest
    assert set(protocol["arms"]) == {"mean", "max", "attention", "grid4"}
    assert "spatial-banks/standard/train/metadata.json" in bundle


def test_pooling_launcher_protocol_records_missing_array_hashes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(
        str(
            Path(__file__).resolve().parents[2]
            / "variants/pixel-repr-ddqn/scripts"
        )
    )
    import colab_pooling_study as launcher

    provenance = tmp_path / "provenance"
    _provenance_tree(provenance)
    array_digest = "ab" * 32
    metadata = {
        "files": {
            "pixels": {"sha256": array_digest},
        },
        "index_sha256": json.loads(
            (provenance / "standard/train/index.json").read_text()
        ),
    }
    (provenance / "standard/train/metadata.json").write_text(json.dumps(metadata))
    _write(provenance / "world.pt", b"world")
    dataset = tmp_path / "dataset"
    _write(dataset / "manifest.json", b"manifest")
    comparison = {
        "world_model_sha256": hashlib.sha256(b"world").hexdigest(),
        "data_sha256": hashlib.sha256(b"manifest").hexdigest(),
        "initial_state_sha256": "c" * 64,
        "frame_index_sha256": "f" * 64,
        "palette_sha256": "e" * 64,
    }
    protocol, bundle = launcher.build_protocol(
        provenance=provenance,
        dataset=dataset,
        run_id="pooling-test",
        spatial_run="spatial-test",
        source_commit="deadbeef",
        comparison=comparison,
    )
    missing = "spatial-banks/standard/train/pixels.npy"
    assert protocol["recovery_targets"] == {missing: array_digest}
    assert protocol["inputs"][missing] == array_digest
    assert missing not in bundle


def test_pooling_launcher_rejects_local_bytes_disagreeing_with_record(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(
        str(
            Path(__file__).resolve().parents[2]
            / "variants/pixel-repr-ddqn/scripts"
        )
    )
    import colab_pooling_study as launcher

    provenance = tmp_path / "provenance"
    _provenance_tree(provenance)
    metadata = {
        "files": {"pixels": {"sha256": "ff" * 32}},
        "index_sha256": "ee" * 32,
    }
    (provenance / "standard/train/metadata.json").write_text(json.dumps(metadata))
    _write(provenance / "standard/train/pixels.npy", b"different-bytes")
    _write(provenance / "world.pt", b"world")
    dataset = tmp_path / "dataset"
    _write(dataset / "manifest.json", b"manifest")
    comparison = {
        "world_model_sha256": hashlib.sha256(b"world").hexdigest(),
        "data_sha256": hashlib.sha256(b"manifest").hexdigest(),
        "initial_state_sha256": "c" * 64,
        "frame_index_sha256": "f" * 64,
        "palette_sha256": "e" * 64,
    }
    with pytest.raises(ValueError, match="disagree with recorded digest"):
        launcher.build_protocol(
            provenance=provenance,
            dataset=dataset,
            run_id="pooling-test",
            spatial_run="spatial-test",
            source_commit="deadbeef",
            comparison=comparison,
        )
