"""Remote phase runner; launched through colab_mvp.py only."""


def run_command(command):
    import subprocess

    print("SETUP", " ".join(command), flush=True)
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait():
        raise RuntimeError(f"Remote command failed: {command[0]}")


def main():
    import hashlib
    import os
    import pathlib
    import sys
    import tarfile

    SOURCE_HASH = os.environ["LEWM_SOURCE_HASH"]
    archive = pathlib.Path("/content/lewm-source.tar.gz")
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == SOURCE_HASH
    root = pathlib.Path("/content/lewm-work")
    root.mkdir(exist_ok=False)
    with tarfile.open(archive) as source:
        source.extractall(root, filter="data")
    os.chdir(root)
    import torch

    assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0), (
        "Colab T4 required"
    )
    print("GPU", torch.cuda.get_device_name(0), flush=True)
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "gymnasium>=1",
            "numpy>=2.4.6",
            "transformers==4.57.6",
            "einops==0.8.2",
            "Pillow",
            "pytest",
        ],
    )
    import shutil

    if not shutil.which("cargo"):
        import urllib.request

        urllib.request.urlretrieve("https://sh.rustup.rs", "/content/rustup-init.sh")
        run_command(["sh", "/content/rustup-init.sh", "-y", "--profile", "minimal"])
    os.environ["PATH"] = (
        str(pathlib.Path.home() / ".cargo/bin") + ":" + os.environ["PATH"]
    )
    run_command(
        [sys.executable, "-m", "pip", "install", "-q", "./native/crates/dodge-python"],
    )
    sys.path.insert(0, str(root / "src"))
    os.environ["PYTHONPATH"] = str(root / "src")
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    run_command(
        [
            sys.executable,
            "-u",
            str(root / "variants/pixel-repr-ddqn/scripts/colab_worker.py"),
        ]
    )


if __name__ == "__main__":
    main()
