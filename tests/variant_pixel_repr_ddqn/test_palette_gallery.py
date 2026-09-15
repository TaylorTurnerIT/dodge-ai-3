from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _gallery():
    path = (
        Path(__file__).parents[2]
        / "variants/pixel-repr-ddqn/scripts/palette_gallery.py"
    )
    spec = importlib.util.spec_from_file_location("palette_gallery", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gallery_preserves_matched_examples_and_rejects_mismatch(tmp_path):
    for prefix in ("plain", "palette-ce", "palette-bce"):
        for mode in ("cls", "projected"):
            run = tmp_path / f"{prefix}-{mode}"
            for step in (512, 2048, 8192):
                images = run / "images" / f"step-{step}"
                images.mkdir(parents=True)
                examples = []
                for split in ("train", "validation"):
                    examples.append(
                        {
                            "split": split,
                            "episode_id": split + "</script>",
                            "frame_index": 3,
                            "recipe_family": "player.stationary",
                        }
                    )
                    for suffix in ("observed", "reconstructed", "wrong-latent"):
                        (images / f"{split}-00-{suffix}.png").write_bytes(b"image")
                (run / f"evaluation-{step}.json").write_text(
                    json.dumps(
                        {"examples": examples, "splits": {}, "wrong_latent_control": {}}
                    )
                )
    gallery = _gallery()
    output = tmp_path / "comparison.html"
    gallery.build(tmp_path, "palette", "plain", output)
    html = output.read_text()
    assert "train</script>" not in html
    assert "train\\u003c/script>" in html
    assert "palette-bce-projected/images/step-8192" in html
    evaluation = tmp_path / "palette-bce-cls/evaluation-512.json"
    payload = json.loads(evaluation.read_text())
    payload["examples"][0]["frame_index"] = 4
    evaluation.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not matched"):
        gallery.build(tmp_path, "palette", "plain", output)
