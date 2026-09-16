"""Restore frozen §Z banks onto a fresh T4 and fit the pooling comparison.

Two modes run as separate fresh processes from the remote driver:

``smoke`` restores and hash-verifies every input (recovering Phase-0-missing
files with the frozen extractors, verifying exact digests, then
re-verifying the promoted bytes in place), checks the shared decoder-core
initialization, and fits a tiny unscored run to scratch outputs proving
milestone evaluation, checkpoint writing, and result serialization work on
this device.

``scored`` re-verifies the complete immutable inputs without any recovery,
re-checks initialization, and fits the four pooling arms at the protocol
milestones.  Fresh heads, optimizers, and sampler states come from the fresh
process; nothing is shared with the smoke run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

CODE_ROOT = Path("/content/lewm-pooling-code")
INPUT_ROOT = Path("/content/lewm-pooling-inputs")
STAGING_ROOT = Path("/content/lewm-pooling-staging")
WORK_ROOT = Path("/content/lewm-pooling-work")
SMOKE_MILESTONES = (4,)


def validate_result(result: dict, protocol: dict) -> None:
    expected = {
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
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"Pooling artifact contract mismatch: {key}")


def _standard_splits_needed(recovery_targets: dict[str, str]) -> list[str]:
    return sorted(
        {
            relpath.split("/")[2]
            for relpath in recovery_targets
            if relpath.startswith("spatial-banks/standard/")
        }
    )


def _spatial_splits_needed(recovery_targets: dict[str, str]) -> list[str]:
    return sorted(
        {
            relpath.split("/")[2]
            for relpath in recovery_targets
            if relpath.startswith("spatial-banks/spatial/")
        }
    )


def _recover_missing(
    protocol: dict,
    inputs: Path,
    staging: Path,
    device: str,
) -> dict[str, Any]:
    """Re-create Phase-0-missing arrays, verify exact digests, promote them."""

    import torch

    from dodge_native_game.variants.pixel_repr_ddqn import large_probe as probe
    from dodge_native_game.variants.pixel_repr_ddqn.pooling_recovery import (
        ENCODE_BATCH_SIZE,
        EXTRACT_BATCH_SIZE,
        FRAMES_PER_EPISODE,
        SAMPLING_SEED,
        verify_files,
    )
    from dodge_native_game.variants.pixel_repr_ddqn.pretrain import load_model
    from dodge_native_game.variants.pixel_repr_ddqn.probe_bank import build_bank
    from dodge_native_game.variants.pixel_repr_ddqn.run_artifacts import file_hash
    from dodge_native_game.variants.pixel_repr_ddqn.spatial_bank import (
        extract_patches,
    )

    recovery_targets = dict(protocol.get("recovery_targets", {}))
    dataset = inputs / "dataset"
    checkpoint = inputs / "world.pt"
    if file_hash(checkpoint) != protocol["inputs"]["world.pt"]:
        raise ValueError("world checkpoint failed verification before recovery")
    model, _ = load_model(checkpoint)
    model = model.to(device).eval()
    model.requires_grad_(False)
    world_hash = protocol["inputs"]["world.pt"]

    banks = inputs / "spatial-banks"
    recovered: list[str] = []
    for split in _standard_splits_needed(recovery_targets):
        # ensure_inputs relocated this split's recorded scaffolding into the
        # staging mirror, so the target is absent and the builders can
        # publish.  Regenerated scaffolding must be byte-identical to the
        # relocated originals before the recovered arrays verify in place.
        target = banks / "standard" / split
        if target.exists():
            raise FileExistsError(f"recovery target already exists: {target}")
        build_bank(
            model,
            dataset,
            target,
            split,
            device=device,
            frames_per_episode=FRAMES_PER_EPISODE,
            seed=SAMPLING_SEED,
            encode_batch_size=ENCODE_BATCH_SIZE,
            checkpoint_sha256=world_hash,
        )
        staged_metadata = (target / "metadata.json").read_bytes()
        original_metadata = (
            staging / f"spatial-banks/standard/{split}/metadata.json"
        ).read_bytes()
        if staged_metadata != original_metadata:
            raise ValueError(
                f"recovered {split} bank metadata differs from recorded metadata"
            )
        wanted = {
            key: value
            for key, value in recovery_targets.items()
            if key.startswith(f"spatial-banks/standard/{split}/")
        }
        verify_files(inputs, wanted)
        recovered.extend(sorted(wanted))

    spatial_needed = _spatial_splits_needed(recovery_targets)
    if spatial_needed:
        bank = probe.open_frame_bank(
            dataset, banks / "standard", checkpoint_sha256=world_hash
        )
        for split in spatial_needed:
            target = banks / "spatial" / split
            if target.exists():
                raise FileExistsError(f"recovery target already exists: {target}")
            extract_patches(
                getattr(bank, split),
                target,
                device=device,
                batch_size=EXTRACT_BATCH_SIZE,
            )
            staged_metadata = (target / "metadata.json").read_bytes()
            original_metadata = (
                staging / f"spatial-banks/spatial/{split}/metadata.json"
            ).read_bytes()
            if staged_metadata != original_metadata:
                raise ValueError(
                    f"recovered {split} sidecar metadata differs from recorded metadata"
                )
            wanted = {
                key: value
                for key, value in recovery_targets.items()
                if key.startswith(f"spatial-banks/spatial/{split}/")
            }
            verify_files(inputs, wanted)
            recovered.extend(sorted(wanted))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    verify_files(inputs, protocol["inputs"])
    return {
        "recovered_files": sorted(recovered),
        "torch_version": str(torch.__version__),
        "device_name": torch.cuda.get_device_name(0),
        "frames_per_episode": FRAMES_PER_EPISODE,
        "sampling_seed": SAMPLING_SEED,
        "encode_batch_size": ENCODE_BATCH_SIZE,
        "extract_batch_size": EXTRACT_BATCH_SIZE,
        "staged_metadata_match": True,
    }


def _relocate_scaffolding(
    inputs: Path,
    staging: Path,
    kinds_splits: set[tuple[str, str]],
    expectations: dict[str, str],
) -> None:
    """Move recorded split scaffolding aside so builders see absent targets.

    ``build_bank``/``extract_patches`` refuse existing output paths and
    publish via ``os.replace``, which fails on FUSE mounts when the
    destination exists.  After digest verification, the recorded
    metadata/index/READY bytes move into the staging mirror; the split
    directory must then be empty (removed here) — anything left over is
    unplanned content and stops the run.  The regenerated scaffolding must
    be byte-identical to the relocated originals (gated in _recover_missing).
    """

    import shutil

    from dodge_native_game.variants.pixel_repr_ddqn.run_artifacts import file_hash

    for kind, split in sorted(kinds_splits):
        for name in ("metadata.json", "index.json", "READY"):
            relpath = f"spatial-banks/{kind}/{split}/{name}"
            if relpath not in expectations:
                continue
            source = inputs / relpath
            if not source.is_file():
                raise ValueError(f"recovery scaffolding missing, stopping: {relpath}")
            if file_hash(source) != expectations[relpath]:
                raise ValueError(f"recovery scaffolding digest mismatch: {relpath}")
            staged = staging / relpath
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(staged))
        try:
            (inputs / "spatial-banks" / kind / split).rmdir()
        except OSError as error:
            raise ValueError(
                "recovery target holds unexpected files: "
                f"spatial-banks/{kind}/{split}"
            ) from error


def ensure_inputs(
    protocol: dict,
    inputs: Path,
    staging: Path,
    *,
    allow_recovery: bool,
) -> dict[str, Any]:
    """Verify the immutable inputs, recovering Phase-0-missing files if allowed."""

    from dodge_native_game.variants.pixel_repr_ddqn.pooling_recovery import (
        missing_files,
        verify_files,
    )

    expectations = dict(protocol["inputs"])
    recovery_targets = dict(protocol.get("recovery_targets", {}))
    absent = missing_files(inputs, expectations)
    unplanned = sorted(set(absent) - set(recovery_targets))
    if unplanned:
        raise ValueError(f"unplanned missing inputs, stopping: {unplanned}")
    planned = {key: recovery_targets[key] for key in sorted(set(absent))}
    if planned and not allow_recovery:
        raise ValueError(
            "inputs incomplete and recovery is not allowed in this mode: "
            + ", ".join(sorted(planned))
        )
    provenance: dict[str, Any] = {"recovered_files": []}
    if planned:
        kinds_splits = set()
        for relpath in planned:
            parts = relpath.split("/")
            kinds_splits.add((parts[1], parts[2]))
        _relocate_scaffolding(inputs, staging, kinds_splits, expectations)
        provenance = _recover_missing(protocol, inputs, staging, "cuda")
    verify_files(inputs, expectations)
    return provenance


def _check_device(protocol: dict) -> None:
    import torch

    if protocol["milestones"] != [512, 2048] or protocol["world_model_updates"] != 0:
        raise ValueError("Invalid pooling budget")
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("T4 required")
    torch.set_num_threads(2)


def _check_core_init(protocol: dict) -> None:
    from dodge_native_game.variants.pixel_repr_ddqn.pooling_readout import (
        make_pooling_decoders,
    )
    from dodge_native_game.variants.pixel_repr_ddqn.probe_bank import _state_digest

    initial = make_pooling_decoders(device="cuda")
    core = initial["mean"].decoder
    if _state_digest(core) != protocol["core_initial_state_sha256"]:
        raise ValueError("Core initialization differs from original spatial study")
    del initial, core


def run_smoke(protocol: dict) -> None:
    import torch

    from dodge_native_game.variants.pixel_repr_ddqn.pooling_probe import run_study
    from dodge_native_game.variants.pixel_repr_ddqn.run_artifacts import atomic_json

    _check_device(protocol)
    provenance = ensure_inputs(protocol, INPUT_ROOT, STAGING_ROOT, allow_recovery=True)
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/variant_pixel_repr_ddqn"],
        cwd=CODE_ROOT,
        check=True,
    )
    _check_core_init(protocol)
    scratch = WORK_ROOT / "scratch-history"
    scratch.mkdir(parents=True, exist_ok=False)
    smoke_id = f"{protocol['run_id']}-smoke"
    run_study(
        INPUT_ROOT / "dataset",
        INPUT_ROOT / "world.pt",
        scratch,
        INPUT_ROOT / "spatial-banks",
        smoke_id,
        device="cuda",
        milestones=SMOKE_MILESTONES,
        eval_scope="validation",
    )
    for mode in protocol["arms"]:
        run = scratch / f"{smoke_id}-{mode}"
        checkpoint = torch.load(run / "decoder-4.pt", weights_only=False)
        if checkpoint.get("step") != 4:
            raise ValueError(f"smoke checkpoint step mismatch: {mode}")
        evaluation = json.loads((run / "evaluation-4.json").read_text())
        if evaluation.get("step") != 4:
            raise ValueError(f"smoke evaluation step mismatch: {mode}")
        report = json.loads((run / "report.json").read_text())
        if report.get("step") != 4:
            raise ValueError(f"smoke report step mismatch: {mode}")
        del checkpoint
    atomic_json(
        WORK_ROOT / "recovery-provenance.json",
        {**provenance, "smoke_milestones": list(SMOKE_MILESTONES)},
    )
    print("SMOKE_COMPLETE", flush=True)


def run_scored(protocol: dict) -> None:
    import torch

    from dodge_native_game.variants.pixel_repr_ddqn.pooling_gallery import (
        write_gallery,
    )
    from dodge_native_game.variants.pixel_repr_ddqn.pooling_probe import run_study
    from dodge_native_game.variants.pixel_repr_ddqn.pooling_readout import (
        make_pooling_decoders,
    )
    from dodge_native_game.variants.pixel_repr_ddqn.run_artifacts import (
        atomic_json,
        file_hash,
    )

    _check_device(protocol)
    ensure_inputs(protocol, INPUT_ROOT, STAGING_ROOT, allow_recovery=False)
    _check_core_init(protocol)
    WORK_ROOT.mkdir(exist_ok=False)
    history = WORK_ROOT / "history"
    history.mkdir()
    initial = make_pooling_decoders(device="cuda")
    torch.save(
        {k: v.cpu() for k, v in initial["mean"].state_dict().items()},
        history / f"{protocol['run_id']}-pooling-initial.pt",
    )
    del initial
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    result = run_study(
        INPUT_ROOT / "dataset",
        INPUT_ROOT / "world.pt",
        history,
        INPUT_ROOT / "spatial-banks",
        protocol["run_id"],
        device="cuda",
        milestones=(512, 2048),
    )
    validate_result(result, protocol)
    write_gallery(history, protocol["run_id"], protocol["source_spatial_run"])
    for relative, expected in protocol["inputs"].items():
        if file_hash(INPUT_ROOT / relative) != expected:
            raise ValueError(f"Frozen input mutated: {relative}")
    recovery_path = WORK_ROOT / "recovery-provenance.json"
    recovery_provenance: dict[str, Any] = (
        json.loads(recovery_path.read_text())
        if recovery_path.is_file()
        else {"recovered_files": []}
    )
    atomic_json(
        history / f"{protocol['run_id']}-environment.json",
        {
            "gpu": torch.cuda.get_device_name(0),
            "torch": str(torch.__version__),
            "source_sha256": os.environ["LEWM_SOURCE_HASH"],
            "protocol": protocol,
            "recovery": recovery_provenance,
            "wheel_source": (
                "cache-hit" if protocol.get("wheel") else "fresh-build"
            ),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    )
    archive = Path("/content/lewm-pooling-results.tar.gz")
    with tarfile.open(archive, "w:gz") as out:
        out.add(history, arcname="history")
    Path("/content/lewm-pooling-results.sha256").write_text(file_hash(archive) + "\n")
    print("POOLING_COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "scored"))
    args = parser.parse_args()
    protocol = json.loads((CODE_ROOT / "pooling_protocol.json").read_text())
    if args.mode == "smoke":
        run_smoke(protocol)
    else:
        run_scored(protocol)


if __name__ == "__main__":
    main()
