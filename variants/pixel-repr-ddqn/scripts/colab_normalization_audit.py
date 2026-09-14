"""Ship a frozen checkpoint and diagnostic source to a new Colab T4."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

from colab_mvp import ROOT, cli


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id.replace("-", "").isalnum():
        raise ValueError("invalid run ID")
    job = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn-audits" / args.run_id
    job.mkdir(parents=True, exist_ok=False)
    protocol = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "data_hash": hashlib.sha256(
            (args.dataset / "manifest.json").read_bytes()
        ).hexdigest(),
        "optimizer_updates": 0,
        "dropout_seeds": [2026, 2027, 2028],
        "calibration": "train-window population moments; disposable copies only",
    }
    (job / "protocol.json").write_text(json.dumps(protocol, indent=2))
    archive = job / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name in (
            "src",
            "native/Cargo.toml",
            "native/Cargo.lock",
            "native/crates",
            "third_party",
            "variants/pixel-repr-ddqn/scripts/colab_normalization_worker.py",
            "tests/variant_pixel_repr_ddqn/test_normalization_audit.py",
        ):
            output.add(
                ROOT / name,
                arcname=name,
                filter=lambda i: None if "__pycache__" in i.name else i,
            )
        output.add(args.checkpoint, arcname="checkpoint.pt")
        output.add(args.dataset, arcname="dataset")
        output.add(job / "protocol.json", arcname="protocol.json")
    if archive.stat().st_size > 1024**3:
        raise ValueError("audit archive exceeds 1GiB")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (job / "source.sha256").write_text(digest + "\n")
    remote = job / "remote.py"
    remote.write_text(
        """import hashlib, os, pathlib, subprocess, sys, tarfile
archive=pathlib.Path('/content/lewm-audit.tar.gz')
assert hashlib.sha256(archive.read_bytes()).hexdigest()==SOURCE_HASH
root=pathlib.Path('/content/lewm-audit');root.mkdir(exist_ok=False)
with tarfile.open(archive) as source: source.extractall(root,filter='data')
import torch
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
print('GPU',torch.cuda.get_device_name(0),flush=True)
subprocess.run([sys.executable,'-m','pip','install','-q','gymnasium>=1','numpy>=2.4.6','transformers==4.57.6','einops==0.8.2','Pillow','pytest'],check=True)
import shutil, urllib.request
if not shutil.which('cargo'):
 urllib.request.urlretrieve('https://sh.rustup.rs','/content/rustup-init.sh')
 subprocess.run(['sh','/content/rustup-init.sh','-y','--profile','minimal'],check=True)
os.environ['PATH']=str(pathlib.Path.home()/'.cargo/bin')+':'+os.environ['PATH']
subprocess.run([sys.executable,'-m','pip','install','-q',str(root/'native/crates/dodge-python')],check=True)
os.environ['PYTHONPATH']=str(root/'src')
os.environ['AUDIT_SOURCE_HASH']=SOURCE_HASH
os.chdir(root)
tests='tests/variant_pixel_repr_ddqn/test_normalization_audit.py'
worker='variants/pixel-repr-ddqn/scripts/colab_normalization_worker.py'
commands=([sys.executable,'-m','pytest',tests,'-q'],[sys.executable,'-u',worker])
for command in commands:
 p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 for line in p.stdout: print(line,end='',flush=True)
 if p.wait(): raise RuntimeError('audit subprocess failed')
""".replace("import hashlib,", f"SOURCE_HASH={digest!r}\nimport hashlib,", 1)
    )
    session = "dodge-" + args.run_id
    cli("new", "--session", session, "--gpu", "T4", timeout=180)
    cli("upload", str(archive), "/content/lewm-audit.tar.gz", "--session", session)
    with (job / "remote.log").open("w") as log:
        process = subprocess.run(
            [
                "colab",
                "--auth",
                "adc",
                "exec",
                "--session",
                session,
                "--file",
                str(remote),
                "--timeout",
                "1800",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=1860,
        )
    if (
        process.returncode
        or "LEWM_NORMALIZATION_AUDIT_COMPLETE" not in (job / "remote.log").read_text()
    ):
        raise RuntimeError(f"audit failed; retained session {session}")
    cli(
        "download",
        "/content/lewm-audit/audit.json",
        str(job / "audit.json"),
        "--session",
        session,
    )
    cli("stop", "--session", session)
    print(f"Audit: {job}", flush=True)


if __name__ == "__main__":
    main()
