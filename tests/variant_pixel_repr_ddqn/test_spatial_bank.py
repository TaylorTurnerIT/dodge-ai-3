"""Real frame-bank metadata and spatial sidecar alignment checks."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from dodge_native_game.variants.pixel_repr_ddqn.probe_bank import ProbeBank, build_bank
from dodge_native_game.variants.pixel_repr_ddqn.spatial_bank import (
    ExpandedFeatures,
    extract_patches,
    open_patches,
)


def test_lazy_controls_preserve_native_patch_pixels():
    palette = np.array([[0, 0, 0], [10, 20, 30], [255, 255, 255]], dtype=np.uint8)
    classes = np.random.default_rng(4).integers(0, 3, (3, 128, 128))
    pixels = palette[classes].transpose(0, 3, 1, 2)
    source = ExpandedFeatures(pixels, palette=palette)
    actual = source[np.array([2, 0])].reshape(2, 16, 16, 3, 8, 8)
    actual = actual.transpose(0, 3, 1, 4, 2, 5).reshape(2, 3, 128, 128)
    assert np.array_equal(actual.argmax(1), classes[[2, 0]])
    assert source[1].shape == (256, 192)
    cls = np.random.default_rng(1).normal(size=(3, 192)).astype("float32")
    assert np.array_equal(ExpandedFeatures(cls)[[2, 0]][:, 123], cls[[2, 0]])


def test_sidecar_actual_producer_and_corruption(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "fixture", Path(__file__).with_name("test_probe_bank.py")
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    dataset = tmp_path / "dataset"
    fixture._make_dataset(dataset, train_count=1, validation_count=1)

    class Model(fixture._FakeModel):
        def encode_readout_tokens(self, pixels):
            cls = self.encode_cls(pixels)
            return cls, torch.arange(256 * 192, dtype=torch.float32).reshape(
                1, 1, 256, 192
            ).expand(len(pixels), 1, -1, -1)

    model = Model()
    build_bank(model, dataset, tmp_path / "standard", "train", device="cpu")
    bank = ProbeBank(tmp_path / "standard")
    local = extract_patches(model, bank, tmp_path / "spatial", device="cpu")
    assert local.shape == (4, 256, 192)
    assert local[0, 1, 0] == 192
    assert np.array_equal(open_patches(bank, tmp_path / "spatial"), local)
    with (tmp_path / "spatial/patches.npy").open("r+b") as output:
        output.seek(-4, 2)
        output.write(b"abcd")
    with pytest.raises(ValueError, match="provenance"):
        open_patches(bank, tmp_path / "spatial")
