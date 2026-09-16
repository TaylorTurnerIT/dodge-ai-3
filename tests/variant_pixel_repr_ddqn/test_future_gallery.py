from __future__ import annotations

import pytest

from dodge_native_game.variants.pixel_repr_ddqn.future_gallery import write_gallery


def test_future_gallery_writes_five_panel_relative_layout(tmp_path) -> None:
    output = write_gallery(tmp_path, "future-screen")
    assert output == tmp_path / "future-screen-comparison.html"
    html = output.read_text()

    assert "__RUN__" not in html
    assert 'run="future-screen"' in html
    assert "repeat(3,minmax(0,1fr))" in html
    assert "grid-template-rows:1fr 1fr" in html
    assert "overflow:hidden" in html
    for kind in (
        "current-observed",
        "current-decoded",
        "next-observed",
        "next-decoded",
        "next-persistence",
        "next-diff",
    ):
        assert kind in html
    assert "palette_changed_per_color_recall" in html
    assert "changed_region_class_error" in html
    assert "Image load error:" in html
    assert "Evaluation load error:" in html
    assert "visualizations.json" in html
    assert "http://" not in html
    assert 'fetch("/' not in html


def test_future_gallery_supports_actual_readout_labels(tmp_path) -> None:
    output = write_gallery(
        tmp_path,
        "actual-screen",
        primary="primary",
        primary_label="actual",
        decode_label="actual latent",
    )
    html = output.read_text()
    assert "__PRIMARY__" not in html
    assert "__DECODE_LABEL__" not in html
    assert 'primary="primary"' in html
    assert '"actual"' in html
    assert "Next decoded (actual latent)" in html


@pytest.mark.parametrize("run_id", ["", "../escape", "has space", "/absolute"])
def test_future_gallery_rejects_invalid_run_ids(tmp_path, run_id: str) -> None:
    with pytest.raises(ValueError, match="invalid run ID"):
        write_gallery(tmp_path, run_id)


def test_future_gallery_rejects_empty_labels(tmp_path) -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        write_gallery(tmp_path, "actual-screen", primary_label="")
