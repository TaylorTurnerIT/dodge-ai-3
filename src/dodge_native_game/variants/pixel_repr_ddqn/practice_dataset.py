"""Import explicit practice captures into a bounded pixel/action corpus."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .collect import (
    MAX_NATIVE_DECISIONS,
    READY_CONTENT,
    READY_MARKER,
    _manifest,
    _record,
)
from .dataset import DatasetValidationError, PixelSequenceDataset, _as_int, _inside
from .practice import PracticeConfig


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _capture(root: Path) -> tuple[bytes, dict[str, Any], bytes, dict[str, Any]]:
    """Read each source once; archived bytes are the bytes we validate."""
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    expected = {
        "schema_version": 1,
        "artifact_format": "pixel-repr-ddqn-practice-v1",
        "variant": "pixel-repr-ddqn",
        "status": "complete",
        "step_frames": 4,
        "observation": {
            "profile": "native-rgb-v1",
            "shape": [3, 128, 128],
            "dtype": "uint8",
        },
    }
    if not isinstance(manifest, dict) or any(
        manifest.get(k) != v for k, v in expected.items()
    ):
        raise DatasetValidationError("invalid practice schema/profile/cadence")
    if type(manifest["schema_version"]) is not int:
        raise DatasetValidationError("invalid practice schema version")
    config = PracticeConfig.from_mapping(manifest["config"])
    if config.to_dict() != manifest["config"] or (
        config.resolved_sha256() != manifest.get("config_sha256")
    ):
        raise DatasetValidationError("practice config hash mismatch")
    if not all(
        _hash(manifest.get(k))
        for k in (
            "generator_sha256",
            "native_module_sha256",
        )
    ):
        raise DatasetValidationError("practice generator/native provenance missing")
    files = manifest.get("files")
    if not isinstance(files, dict) or "episode" not in files:
        raise DatasetValidationError("practice files missing")
    blobs = {}
    for key, record in files.items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise DatasetValidationError("invalid practice file record")
        data = _inside(root, record["path"]).read_bytes()
        if _digest(data) != record.get("sha256") or len(data) != record.get("bytes"):
            raise DatasetValidationError("practice file hash/size mismatch")
        blobs[key] = data
    source_hash = manifest.get("source_sha256")
    if source_hash is not None and (
        not _hash(source_hash) or _digest(blobs.get("source", b"")) != source_hash
    ):
        raise DatasetValidationError("practice source hash mismatch")
    episode = blobs["episode"]
    with np.load(io.BytesIO(episode), allow_pickle=False) as archive:
        if set(archive.files) != {"pixels", "actions", "terminated", "truncated"}:
            raise DatasetValidationError("unexpected practice episode fields")
        arrays = {key: archive[key] for key in archive.files}
    actions = arrays["actions"]
    count = _as_int(manifest.get("decision_count"), "decision_count")
    if not 1 <= count <= config.max_decisions:
        raise DatasetValidationError("practice decision count outside config budget")
    trace = manifest.get("action_trace")
    if not isinstance(trace, list) or any(type(a) is not int for a in trace):
        raise DatasetValidationError("invalid practice action trace")
    if actions.shape != (count,) or actions.tolist() != trace:
        raise DatasetValidationError("practice action trace/count mismatch")
    for key in ("terminated", "truncated", "finished", "failed", "success"):
        if type(manifest.get(key)) is not bool:
            raise DatasetValidationError("practice outcome flags must be boolean")
    terminal, truncated = manifest["terminated"], manifest["truncated"]
    finished, failed = manifest["finished"], manifest["failed"]
    if terminal == truncated or sum((terminal, finished, failed)) > 1:
        raise DatasetValidationError("incompatible practice outcome flags")
    success = finished and not terminal and not failed
    if manifest["success"] != success or manifest.get("failure") != (
        None if success else "terminated" if terminal else "timeout"
    ):
        raise DatasetValidationError("practice success/failure mismatch")
    for key in ("terminated", "truncated"):
        values = arrays[key]
        if (
            values.dtype != np.bool_
            or values.shape != (count,)
            or (bool(values[-1]) != manifest[key] or values[:-1].any())
        ):
            raise DatasetValidationError("practice episode boundary mismatch")
    arrays.update(terminal=terminal, capped=truncated)
    return manifest_bytes, manifest, episode, arrays


def import_practice_dataset(
    output: Path,
    train: list[Path],
    validation: list[Path],
) -> dict[str, Any]:
    """Publish only after source, split and sequence validation succeeds."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"dataset output exists: {output}")
    if not train or not validation:
        raise DatasetValidationError("both practice splits require captures")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(f".{output.name}.lock")
    lock.mkdir()
    staging = None
    try:
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent)
        )
        records: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
        seeds: set[int] = set()
        hashes: dict[str, str] = {}
        modes: set[bool] = set()
        total = 0
        for split, sources in (("train", train), ("validation", validation)):
            for index, source in enumerate(sources):
                raw, source_manifest, episode, arrays = _capture(Path(source))
                seed = _as_int(source_manifest.get("seed"), "practice seed")
                if seed in seeds:
                    raise DatasetValidationError("duplicate practice seed")
                seeds.add(seed)
                digest = _digest(episode)
                if digest in hashes and hashes[digest] != split:
                    raise DatasetValidationError("identical episodes across splits")
                hashes[digest] = split
                modes.add(source_manifest["config"]["invulnerable"])
                if len(modes) > 1:
                    raise DatasetValidationError("mixed invulnerable/lethal practice")
                total += len(arrays["actions"])
                if total > MAX_NATIVE_DECISIONS:
                    raise DatasetValidationError("practice corpus exceeds decision cap")
                relative = Path("episodes") / split / f"episode-{index:06d}.npz"
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(episode)
                provenance = relative.with_suffix(".manifest.json")
                (staging / provenance).write_bytes(raw)
                record = _record(
                    split=split,
                    index=index,
                    game_seed=seed,
                    path=relative,
                    file_path=destination,
                    arrays=arrays,
                )
                record["practice"] = {
                    "manifest_path": provenance.as_posix(),
                    "manifest_sha256": _digest(raw),
                    **{
                        key: source_manifest[key]
                        for key in (
                            "config_sha256",
                            "generator_sha256",
                            "native_module_sha256",
                        )
                    },
                }
                records[split].append(record)
        manifest = _manifest(
            records=records,
            train_seeds=[r["seed"] for r in records["train"]],
            validation_seeds=[r["seed"] for r in records["validation"]],
            max_steps=max(
                r["transition_count"] for rows in records.values() for r in rows
            ),
            collection_seed=0,
            scenario={},
        )
        manifest.pop("scenario")
        manifest["collection"].update(
            policy="scripted-practice-import-v1", seed=None, new_native_decisions=0
        )
        manifest["practice_import"] = {
            "importer_sha256": _digest(Path(__file__).read_bytes()),
            "invulnerable": modes.pop(),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        (staging / READY_MARKER).write_text(READY_CONTENT)
        for split in records:
            if not len(PixelSequenceDataset(staging, split=split, history_size=3)):
                raise DatasetValidationError("both splits need history3 windows")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"dataset output appeared: {output}")
        os.rename(staging, output)
        staging = None
        return manifest
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        lock.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    result = import_practice_dataset(args.output, args.train, args.validation)
    print(json.dumps({"dataset_id": result["dataset_id"], "splits": result["splits"]}))


if __name__ == "__main__":
    main()
