from __future__ import annotations

import pytest

from dodge_native_game.variants.pixel_repr_ddqn.pooling_gallery import write_gallery


def test_pooling_gallery_writes_four_matched_relative_panels(tmp_path) -> None:
    output = write_gallery(tmp_path, "pooling-screen", "spatial-screen")
    assert output == tmp_path / "pooling-screen-comparison.html"
    html = output.read_text()

    assert "__POOLING_RUN__" not in html
    assert "__SPATIAL_RUN__" not in html
    assert 'poolingRun="pooling-screen"' in html
    assert 'spatialRun="spatial-screen"' in html
    assert "repeat(4,minmax(0,1fr))" in html
    assert "overflow:hidden" in html
    assert "changed_region_mse" in html
    assert "palette_changed_per_color_recall" in html
    assert "Image load error:" in html
    assert "Evaluation load error:" in html
    assert "path(`${poolingRun}-${mode}`" in html
    assert 'modes=["mean","max","attention","grid4"]' in html
    for arm in ("spatial-screen-patch", "spatial-screen-cls"):
        assert f"path(`${{spatialRun}}-{arm.rsplit('-', 1)[-1]}`" in html
    assert "http://" not in html
    assert 'fetch("/' not in html


@pytest.mark.parametrize("run_id", ["", "../escape", "has space", "/absolute"])
def test_pooling_gallery_rejects_invalid_run_ids(tmp_path, run_id: str) -> None:
    with pytest.raises(ValueError, match="invalid"):
        write_gallery(tmp_path, run_id, "spatial-screen")
    with pytest.raises(ValueError, match="invalid"):
        write_gallery(tmp_path, "pooling-screen", run_id)
