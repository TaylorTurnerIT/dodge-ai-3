from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_worker_bank_audit_paths_are_source_sidecars(tmp_path, monkeypatch) -> None:
    scripts = Path(__file__).resolve().parents[2] / "variants/pixel-repr-ddqn/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    worker = _load_script(scripts / "colab_spatial_worker.py", "spatial_worker")

    bank_root, required = worker._bank_artifacts(tmp_path)

    assert bank_root == tmp_path / "spatial-banks"
    assert required
    assert all(path.is_relative_to(bank_root) for path in required)
    assert all("history" not in path.parts for path in required)
    assert required[-2:] == (
        bank_root / "spatial" / "validation" / "patches.npy",
        bank_root / "standard" / "validation" / "cls.npy",
    )


def test_spatial_protocol_matches_worker_contract(monkeypatch) -> None:
    scripts = Path(__file__).resolve().parents[2] / "variants/pixel-repr-ddqn/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    launcher = _load_script(scripts / "colab_spatial_study.py", "spatial_launcher")
    worker = _load_script(
        scripts / "colab_spatial_worker.py", "spatial_worker_contract"
    )

    protocol = launcher._protocol(
        checkpoint_sha256="a" * 64,
        data_sha256="b" * 64,
        run_id="spatial-test",
        source_commit="fixture",
    )
    worker._require_protocol(protocol)
