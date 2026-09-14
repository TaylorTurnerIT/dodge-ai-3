"""Fresh-process frozen checkpoint audit; no optimizer or native collection."""


def main():
    import hashlib
    import json
    import os
    from pathlib import Path

    import torch

    from dodge_native_game.variants.pixel_repr_ddqn.dataset import PixelSequenceDataset
    from dodge_native_game.variants.pixel_repr_ddqn.normalization_audit import audit
    from dodge_native_game.variants.pixel_repr_ddqn.pretrain import load_model

    assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0)
    torch.set_num_threads(2)
    root = Path.cwd()
    protocol = json.loads((root / "protocol.json").read_text())

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    checkpoint = root / "checkpoint.pt"
    before = digest(checkpoint)
    assert before == protocol["checkpoint_sha256"]
    assert digest(root / "dataset/manifest.json") == protocol["data_hash"]
    model, payload = load_model(checkpoint)
    assert payload["data_hash"] == protocol["data_hash"]
    model = model.to("cuda").eval().requires_grad_(False)
    data = {}
    for split in ("train", "validation"):
        ds = PixelSequenceDataset(root / "dataset", split=split)
        data[split] = (
            torch.stack([ds[i]["pixels"] for i in range(len(ds))]).cuda(),
            torch.stack([ds[i]["actions"] for i in range(len(ds))]).cuda(),
        )
    # Recover the final optimizer batch's sampling indices without running it.
    rng = torch.Generator().manual_seed(payload["seed"] + 1)
    for _ in range(payload["step"]):
        indices = torch.randint(
            len(data["train"][0]), (payload["batch_size"],), generator=rng
        )
    assert torch.equal(rng.get_state(), payload["sampling_rng"])
    print(
        "Frozen checkpoint loaded; running factorial and buffer-copy diagnostics",
        flush=True,
    )
    result = audit(model, *data["train"], *data["validation"], indices)
    assert before == digest(checkpoint)
    result.update(
        protocol=protocol,
        checkpoint_unchanged=True,
        gpu=torch.cuda.get_device_name(0),
        torch=str(torch.__version__),
        source_sha256=os.environ["AUDIT_SOURCE_HASH"],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
    (root / "audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print("LEWM_NORMALIZATION_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
