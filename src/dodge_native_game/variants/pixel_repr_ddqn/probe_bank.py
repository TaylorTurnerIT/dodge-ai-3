"""Bounded, disk-backed current-frame features for frozen decoder probes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .large_dataset import LARGE_OBSERVATION_SHAPE, _read_episode, read_dataset_metadata
from .run_artifacts import atomic_json, file_hash

BANK_FORMAT = "pixel-repr-ddqn-probe-bank-v1"
READY_CONTENT = f"{BANK_FORMAT}\n"
FRAME_FIRST = 1
FRAME_LAST = 128
MAX_FRAMES_PER_EPISODE = 4
DEFAULT_FRAMES_PER_EPISODE = 4
DEFAULT_SEED = 903
DEFAULT_ENCODE_BATCH_SIZE = 32
_REPRESENTATIONS = ("cls", "projected")


def _positive_int(value: object, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1 or (maximum is not None and result > maximum):
        bound = f" in 1..{maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be{bound}")
    return result


def _hash_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return value


def _state_digest(model: torch.nn.Module) -> str:
    """Hash tensor values and their names without retaining a second state dict."""

    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"model state entry {name!r} is not a tensor")
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _restore_modes(model: torch.nn.Module, modes: list[bool]) -> None:
    for module, mode in zip(model.modules(), modes, strict=True):
        module.training = mode


def _select_frames(
    record_count: int, frames_per_episode: int, rng: np.random.Generator
) -> np.ndarray:
    if record_count != FRAME_LAST:
        raise ValueError(f"published episodes must have {FRAME_LAST} transitions")
    selected = rng.choice(
        np.arange(FRAME_FIRST, FRAME_LAST + 1, dtype=np.int64),
        size=frames_per_episode,
        replace=False,
    )
    return np.sort(selected).astype(np.int64, copy=False)


def _encode(
    model: torch.nn.Module,
    pixels: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    batch = torch.from_numpy(np.ascontiguousarray(pixels)).unsqueeze(1).to(device)
    with torch.no_grad():
        if hasattr(model, "encode_cls"):
            cls = model.encode_cls(batch)
        else:
            cls = model.encode_representation(batch, representation="cls")
        if (
            hasattr(model, "_apply_projector")
            and hasattr(model, "projector")
        ):
            projected = model._apply_projector(model.projector, cls)
        else:
            projected = model.encode_representation(
                batch, representation="projected"
            )
    for name, value in (("cls", cls), ("projected", projected)):
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise ValueError(f"{name} encoder output must have shape (B,1,D)")
        if value.shape[0] != pixels.shape[0] or value.shape[1] != 1:
            raise ValueError(f"{name} encoder output must have shape (B,1,D)")
        if not torch.is_floating_point(value):
            raise TypeError(f"{name} encoder output must be floating point")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} encoder output contains nonfinite values")
    cls_cpu = cls[:, 0].detach().to(device="cpu", dtype=torch.float32).numpy()
    projected_cpu = (
        projected[:, 0].detach().to(device="cpu", dtype=torch.float32).numpy()
    )
    return cls_cpu, projected_cpu


def _write_ready(path: Path) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".READY-", dir=path)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(READY_CONTENT)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path / "READY")
    finally:
        Path(temporary).unlink(missing_ok=True)


def _verify_maps(root: Path, *, count: int, dimension: int) -> None:
    expected = {
        "pixels.npy": (np.uint8, (count, *LARGE_OBSERVATION_SHAPE)),
        "changed.npy": (
            np.bool_,
            (count, LARGE_OBSERVATION_SHAPE[1], LARGE_OBSERVATION_SHAPE[2]),
        ),
        "cls.npy": (np.float32, (count, dimension)),
        "projected.npy": (np.float32, (count, dimension)),
    }
    for filename, (dtype, shape) in expected.items():
        mapped = np.lib.format.open_memmap(root / filename, mode="r")
        if mapped.dtype != dtype or mapped.shape != shape or mapped.flags.writeable:
            raise RuntimeError(f"probe bank map {filename} has invalid shape or dtype")
        if filename in {"cls.npy", "projected.npy"}:
            for start in range(0, count, DEFAULT_ENCODE_BATCH_SIZE):
                values = mapped[start : start + DEFAULT_ENCODE_BATCH_SIZE]
                if not np.isfinite(values).all():
                    raise RuntimeError(
                        f"probe bank map {filename} contains nonfinite values"
                    )
        del mapped


def build_bank(
    model: torch.nn.Module,
    dataset_root: Path,
    output: Path,
    split: str,
    device: str = "cuda",
    frames_per_episode: int = DEFAULT_FRAMES_PER_EPISODE,
    seed: int = DEFAULT_SEED,
    encode_batch_size: int = DEFAULT_ENCODE_BATCH_SIZE,
    checkpoint_sha256: str | None = None,
) -> Path:
    """Publish one split's matched CLS/projected feature bank atomically."""

    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")
    frames_per_episode = _positive_int(
        frames_per_episode, "frames_per_episode", maximum=MAX_FRAMES_PER_EPISODE
    )
    encode_batch_size = _positive_int(
        encode_batch_size, "encode_batch_size", maximum=DEFAULT_ENCODE_BATCH_SIZE
    )
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if checkpoint_sha256 is not None:
        checkpoint_sha256 = _hash_digest(checkpoint_sha256, "checkpoint_sha256")

    dataset_root = Path(dataset_root)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"probe bank output already exists: {output}")
    dataset = read_dataset_metadata(dataset_root)
    records = dataset.records[split]
    dataset_hash = file_hash(dataset_root / "manifest.json")
    source_hash = file_hash(Path(__file__))
    before = _state_digest(model)
    selected: list[tuple[str, int]] = []
    rng = np.random.default_rng(int(seed))
    for record in records:
        selected.extend(
            (record.episode_id, int(frame))
            for frame in _select_frames(record.count, frames_per_episode, rng)
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.pending-", dir=output.parent)
    )
    modes = [module.training for module in model.modules()]
    model.eval()
    maps: dict[str, np.memmap] = {}
    try:
        pending_pixels: list[np.ndarray] = []
        pending_changed: list[np.ndarray] = []
        pending_positions: list[int] = []
        dimension: int | None = None
        write_position = 0

        def flush_pending() -> None:
            nonlocal dimension, write_position, maps
            if not pending_pixels:
                return
            encoded_cls, encoded_projected = _encode(
                model, np.stack(pending_pixels), device=torch.device(device)
            )
            if encoded_cls.shape != encoded_projected.shape:
                raise ValueError("CLS and projected feature shapes differ")
            if dimension is None:
                dimension = int(encoded_cls.shape[1])
                shape_count = len(selected)
                maps = {
                    "pixels": np.lib.format.open_memmap(
                        temporary / "pixels.npy",
                        mode="w+",
                        dtype=np.uint8,
                        shape=(shape_count, *LARGE_OBSERVATION_SHAPE),
                    ),
                    "changed": np.lib.format.open_memmap(
                        temporary / "changed.npy",
                        mode="w+",
                        dtype=np.bool_,
                        shape=(shape_count, 128, 128),
                    ),
                    "cls": np.lib.format.open_memmap(
                        temporary / "cls.npy",
                        mode="w+",
                        dtype=np.float32,
                        shape=(shape_count, dimension),
                    ),
                    "projected": np.lib.format.open_memmap(
                        temporary / "projected.npy",
                        mode="w+",
                        dtype=np.float32,
                        shape=(shape_count, dimension),
                    ),
                }
            elif encoded_cls.shape[1] != dimension:
                raise ValueError("encoder feature dimension changed during bank build")
            batch_count = len(pending_pixels)
            positions = np.asarray(pending_positions, dtype=np.int64)
            expected = np.arange(write_position, write_position + batch_count)
            if not np.array_equal(positions, expected):
                raise RuntimeError("probe bank writer lost sample order")
            maps["pixels"][positions] = np.stack(pending_pixels)
            maps["changed"][positions] = np.stack(pending_changed)
            maps["cls"][positions] = encoded_cls
            maps["projected"][positions] = encoded_projected
            write_position += batch_count
            pending_pixels.clear()
            pending_changed.clear()
            pending_positions.clear()

        position = 0
        for episode_number, record in enumerate(records, start=1):
            episode_pixels, _ = _read_episode(record, verify_hash=True)
            frames = [
                frame
                for _, frame in selected[position : position + frames_per_episode]
            ]
            for frame in frames:
                pending_pixels.append(episode_pixels[frame].copy())
                pending_changed.append(
                    np.any(episode_pixels[frame] != episode_pixels[frame - 1], axis=0)
                )
                pending_positions.append(position)
                position += 1
                if len(pending_pixels) >= encode_batch_size:
                    flush_pending()
            del episode_pixels
            if episode_number % 128 == 0:
                print(f"probe bank {split}: {episode_number} episodes", flush=True)
        flush_pending()
        if dimension is None or write_position != len(selected):
            raise RuntimeError("probe bank contains no encoded frames")
        for mapped in maps.values():
            mapped.flush()
        maps.clear()

        _verify_maps(temporary, count=len(selected), dimension=dimension)
        index = [
            {"episode_id": episode_id, "frame": frame}
            for episode_id, frame in selected
        ]
        atomic_json(temporary / "index.json", index)
        index_hash = file_hash(temporary / "index.json")
        map_hashes = {
            filename: file_hash(temporary / filename)
            for filename in ("pixels.npy", "changed.npy", "cls.npy", "projected.npy")
        }
        metadata: dict[str, Any] = {
            "format": BANK_FORMAT,
            "schema_version": 1,
            "split": split,
            "seed": int(seed),
            "frames_per_episode": frames_per_episode,
            "max_frames_per_episode": MAX_FRAMES_PER_EPISODE,
            "episode_count": len(records),
            "frame_count": len(selected),
            "observation_shape": list(LARGE_OBSERVATION_SHAPE),
            "observation_dtype": "uint8",
            "feature_dtype": "float32",
            "feature_dim": dimension,
            "data_hash": dataset_hash,
            "dataset_manifest_sha256": dataset_hash,
            "checkpoint_sha256": checkpoint_sha256 or before,
            "source_sha256": source_hash,
            "index_sha256": index_hash,
            "files": {
                "pixels": {
                    "shape": [len(selected), *LARGE_OBSERVATION_SHAPE],
                    "dtype": "uint8",
                    "sha256": map_hashes["pixels.npy"],
                },
                "changed": {
                    "shape": [len(selected), 128, 128],
                    "dtype": "bool",
                    "sha256": map_hashes["changed.npy"],
                },
                "cls": {
                    "shape": [len(selected), dimension],
                    "dtype": "float32",
                    "sha256": map_hashes["cls.npy"],
                },
                "projected": {
                    "shape": [len(selected), dimension],
                    "dtype": "float32",
                    "sha256": map_hashes["projected.npy"],
                },
            },
        }
        atomic_json(temporary / "metadata.json", metadata)
        if json.loads((temporary / "index.json").read_text()) != index:
            raise RuntimeError("probe bank index verification failed")
        if json.loads((temporary / "metadata.json").read_text()) != metadata:
            raise RuntimeError("probe bank metadata verification failed")
        if file_hash(dataset_root / "manifest.json") != dataset_hash:
            raise RuntimeError("dataset manifest changed during bank build")
        after = _state_digest(model)
        if after != before:
            raise RuntimeError("frozen model changed during bank build")
        _write_ready(temporary)
        os.replace(temporary, output)
        temporary = Path()
        return output
    finally:
        for mapped in maps.values():
            mapped.flush()
        if temporary != Path():
            shutil.rmtree(temporary, ignore_errors=True)
        _restore_modes(model, modes)


class ProbeBank:
    """Read-only view of a published split bank."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        if (self.root / "READY").read_text() != READY_CONTENT:
            raise ValueError("probe bank has no valid final READY marker")
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        self.index = json.loads((self.root / "index.json").read_text())
        if self.metadata.get("format") != BANK_FORMAT or not isinstance(
            self.index, list
        ):
            raise ValueError("invalid probe bank metadata")
        expected = len(self.index)
        frame_count = self.metadata.get("frame_count")
        if (
            isinstance(frame_count, bool)
            or not isinstance(frame_count, int)
            or expected != frame_count
        ):
            raise ValueError("probe bank index and metadata counts disagree")
        if file_hash(self.root / "index.json") != self.metadata.get("index_sha256"):
            raise ValueError("probe bank index hash mismatch")
        dimension = self.metadata.get("feature_dim")
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
        ):
            raise ValueError("probe bank feature dimension is invalid")
        self.pixels = np.lib.format.open_memmap(self.root / "pixels.npy", mode="r")
        self.changed = np.lib.format.open_memmap(self.root / "changed.npy", mode="r")
        self.cls = np.lib.format.open_memmap(self.root / "cls.npy", mode="r")
        self.projected = np.lib.format.open_memmap(
            self.root / "projected.npy", mode="r"
        )
        expected_shapes = {
            "pixels": (expected, *LARGE_OBSERVATION_SHAPE),
            "changed": (expected, 128, 128),
            "cls": (expected, dimension),
            "projected": (expected, dimension),
        }
        actual = {
            "pixels": self.pixels,
            "changed": self.changed,
            "cls": self.cls,
            "projected": self.projected,
        }
        expected_dtypes = {
            "pixels": np.uint8,
            "changed": np.bool_,
            "cls": np.float32,
            "projected": np.float32,
        }
        file_metadata = self.metadata.get("files")
        if not isinstance(file_metadata, dict):
            raise ValueError("probe bank file metadata is missing")
        for name, mapped in actual.items():
            if (
                mapped.shape != expected_shapes[name]
                or mapped.dtype != expected_dtypes[name]
            ):
                raise ValueError(f"probe bank {name} map shape or dtype is invalid")
            declaration = file_metadata.get(name)
            if not isinstance(declaration, dict):
                raise ValueError(f"probe bank {name} metadata is missing")
            if declaration.get("shape") != list(expected_shapes[name]):
                raise ValueError(f"probe bank {name} metadata shape is invalid")
            if declaration.get("dtype") != np.dtype(expected_dtypes[name]).name:
                raise ValueError(f"probe bank {name} metadata dtype is invalid")
        if any(set(item) != {"episode_id", "frame"} for item in self.index):
            raise ValueError("probe bank index contains unsupported fields")

    def __len__(self) -> int:
        return int(self.pixels.shape[0])

    def fetch(
        self,
        indices: object,
        representation: str,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if representation not in _REPRESENTATIONS:
            raise ValueError("representation must be 'cls' or 'projected'")
        if isinstance(indices, torch.Tensor):
            raw = indices.detach().cpu().numpy()
        else:
            raw = np.asarray(indices)
        if raw.ndim == 0:
            raw = raw.reshape(1)
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise TypeError("indices must be a one-dimensional integer sequence")
        positions = raw.astype(np.int64, copy=False)
        if np.any(positions < 0) or np.any(positions >= len(self)):
            raise IndexError("probe bank index out of range")
        feature_map = self.cls if representation == "cls" else self.projected
        z = torch.from_numpy(np.asarray(feature_map[positions]).copy()).to(
            device=device, dtype=torch.float32
        )
        pixels = torch.from_numpy(np.asarray(self.pixels[positions]).copy()).to(
            device=device, dtype=torch.float32
        )
        pixels.div_(255.0)
        changed = torch.from_numpy(np.asarray(self.changed[positions]).copy()).to(
            device=device, dtype=torch.bool
        )
        return z, pixels, changed


__all__ = ["ProbeBank", "build_bank"]
