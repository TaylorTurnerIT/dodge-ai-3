"""Reuse the spatial study T4 for a separately frozen pooling experiment."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from colab_large_probe import ROOT, _validate_run_id, cli, digest


def refresh(session: str, endpoint: str):
    executable = (
        Path(shutil.which("colab")).read_text().splitlines()[0].removeprefix("#!")
    )
    subprocess.run(
        [
            executable,
            str(Path(__file__).with_name("colab_refresh.py")),
            "--session",
            session,
            "--endpoint",
            endpoint,
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--spatial-run", required=True)
    args = parser.parse_args()
    for value in [args.run_id, args.spatial_run]:
        _validate_run_id(value, "run ID")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("Freeze and commit source before fitting")
    history = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn"
    provenance = history / f"{args.spatial_run}-bank-provenance"
    comparison = json.loads(
        (history / f"{args.spatial_run}-spatial-comparison.json").read_text()
    )
    inputs = {
        "world.pt": comparison["world_model_sha256"],
        "dataset/manifest.json": comparison["data_sha256"],
    }
    for kind in ["standard", "spatial"]:
        for split in ["train", "validation"]:
            path = provenance / kind / split
            for name in (
                ["metadata.json", "index.json", "READY"]
                if kind == "standard"
                else ["metadata.json", "READY"]
            ):
                inputs[f"spatial-banks/{kind}/{split}/{name}"] = digest(path / name)
            if kind == "spatial":
                inputs[f"spatial-banks/{kind}/{split}/patches.npy"] = json.loads(
                    (path / "metadata.json").read_text()
                )["patches_sha256"]
    protocol = {
        "run_id": args.run_id,
        "source_spatial_run": args.spatial_run,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "milestones": [512, 2048],
        "world_model_updates": 0,
        "inputs": inputs,
        "core_initial_state_sha256": comparison["initial_state_sha256"],
        "frame_index_sha256": comparison["frame_index_sha256"],
        "palette_sha256": comparison["palette_sha256"],
        "arms": ["mean", "max", "attention", "grid4"],
        "worker_timeout_seconds": 7200,
    }
    job = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn-jobs" / args.run_id
    job.mkdir(exist_ok=False)
    (job / "pooling_protocol.json").write_text(json.dumps(protocol, indent=2))
    archive = job / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as out:
        for name in [
            "src",
            "native/Cargo.toml",
            "native/Cargo.lock",
            "native/crates",
            "third_party",
            "variants/pixel-repr-ddqn",
            "tests/variant_pixel_repr_ddqn",
            "references/manifest.json",
            "references/le-wm/module.py",
            "references/paper/lewm-v3.md",
            "references/pooling",
        ]:
            out.add(
                ROOT / name,
                arcname=name,
                filter=lambda m: None if "__pycache__" in m.name else m,
            )
        out.add(job / "pooling_protocol.json", arcname="pooling_protocol.json")
    source_hash = digest(archive)
    (job / "source.sha256").write_text(source_hash + "\n")
    refresh(args.session, args.endpoint)
    # Unique upload paths preserve the previous experiment's archive.
    count = 0
    with archive.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            part = job / f"part-{count:03d}"
            part.write_bytes(block)
            cli(
                "upload",
                str(part),
                f"/content/lewm-pooling.part-{count:03d}",
                "--session",
                args.session,
            )
            count += 1
    remote = job / "remote.py"
    remote.write_text(f"""import os,sys,hashlib,tarfile,subprocess
from pathlib import Path
archive=Path('/content/lewm-pooling-source.tar.gz')
with archive.open('wb') as output:
 for i in range({count}):
  part=Path(f'/content/lewm-pooling.part-{{i:03d}}')
  output.write(part.read_bytes())
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={source_hash!r}
root=Path('/content/lewm-pooling-code');root.mkdir(exist_ok=False)
with tarfile.open(archive) as bundle: bundle.extractall(root,filter='data')
env=dict(os.environ,PYTHONPATH=str(root/'src'),LEWM_SOURCE_HASH={source_hash!r})
subprocess.run([sys.executable,str(root/'variants/pixel-repr-ddqn/scripts/colab_pooling_worker.py')],cwd=root,env=env,check=True,timeout=7100)
""")
    with (job / "remote.log").open("w") as log:
        result = subprocess.run(
            [
                "colab",
                "--auth",
                "adc",
                "exec",
                "--session",
                args.session,
                "--file",
                str(remote),
                "--timeout",
                "7200",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(
            "Pooling worker failed; original T4 retained, inspect remote.log"
        )
    refresh(args.session, args.endpoint)
    for remote_name, local_name in [
        ("lewm-pooling-results.sha256", "results.sha256"),
        ("lewm-pooling-results.tar.gz", "results.tar.gz"),
    ]:
        try:
            cli(
                "download",
                f"/content/{remote_name}",
                str(job / local_name),
                "--session",
                args.session,
                timeout=600,
            )
        except RuntimeError:
            refresh(args.session, args.endpoint)
            cli(
                "download",
                f"/content/{remote_name}",
                str(job / local_name),
                "--session",
                args.session,
                timeout=600,
            )
    if digest(job / "results.tar.gz") != (job / "results.sha256").read_text().strip():
        raise RuntimeError("Result archive checksum mismatch")
    with tarfile.open(job / "results.tar.gz") as bundle:
        bundle.extractall(job / "results", filter="data")
    shutil.copytree(job / "results/history", history, dirs_exist_ok=True)
    print("Pooling archive verified and retrieved; T4 retained for parent audit")


if __name__ == "__main__":
    main()
