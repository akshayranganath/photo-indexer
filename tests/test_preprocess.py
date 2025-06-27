"""
tests/test_preprocess.py
~~~~~~~~~~~~~~~~~~~~~~~~

Unit-tests for the thumbnail / EXIF pre-processing layer
(`src/photo_indexer/pipelines/preprocess.py`).

The tests assume:

* a **sample RAW file** lives at ``tests/data/sample1.NEF`` (≈ 2 – 3 MB).
  –––► You can replace it with any tiny NEF from your camera; the file is
        git-ignored so the repo stays small.
* the public API of *preprocess.py* exposes

      load_nef(path: Path) -> Tuple[np.ndarray, dict]
      write_thumbnail(img: np.ndarray, thumb_path: Path, size: int = 512) -> None
      preprocess_file(path: Path, thumb_root: Path, size: int = 512) -> PreprocessResult

  where ``PreprocessResult`` is a `dataclass` holding at least
  ``rgb``, ``exif`` and ``thumb_path`` attributes.
"""

from __future__ import annotations

import pathlib
import shutil

import numpy as np
import pytest

try:
    import rawpy  # noqa: F401  # skip tests gracefully if rawpy isn't present
except ModuleNotFoundError:
    pytest.skip("rawpy not installed – skipping pre-processing tests", allow_module_level=True)

from PIL import Image

from photo_indexer.pipelines import preprocess


# ---------------------------------------------------------------------------
# Paths & fixtures
# ---------------------------------------------------------------------------

HERE = pathlib.Path(__file__).parent
SAMPLE_NEF = HERE / "data" / "sample1.NEF"


@pytest.fixture(scope="session")
def tmp_thumb_root(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Ephemeral folder for thumbnails that survives the whole test session."""
    return tmp_path_factory.mktemp("thumbs")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not SAMPLE_NEF.exists(), reason="Provide a tiny sample NEF in tests/data/ to run this suite."
)
def test_load_nef_returns_rgb_and_exif() -> None:
    """load_nef should demosaic the RAW and deliver basic EXIF."""
    rgb, exif = preprocess.load_nef(SAMPLE_NEF)

    # RGB image sanity
    assert isinstance(rgb, np.ndarray), "RGB output must be a NumPy array"
    assert rgb.ndim == 3 and rgb.shape[2] == 3, "RGB array must have shape (h, w, 3)"
    assert rgb.dtype == np.uint8, "Thumbnail pipeline expects 8-bit data"

    # EXIF sanity
    assert "DateTimeOriginal" in exif, "DateTimeOriginal missing from EXIF dict"
    assert exif["DateTimeOriginal"]  # non-empty


def test_write_thumbnail_creates_resized_jpeg(tmp_path: pathlib.Path) -> None:
    """write_thumbnail should emit a JPEG whose longest edge == size."""
    # fake 4:3 test image (PIL generates quicker than rawpy)
    img = (np.random.rand(600, 800, 3) * 255).astype("uint8")
    thumb_path = tmp_path / "thumb.jpg"

    preprocess.write_thumbnail(img, thumb_path, size=256)

    assert thumb_path.exists(), "Thumbnail file not written"

    with Image.open(thumb_path) as im:
        assert max(im.size) == 256, "Thumbnail longest edge != 256 px"
        assert im.format == "JPEG", "Output must be JPEG"


@pytest.mark.skipif(
    not SAMPLE_NEF.exists(), reason="Provide a tiny sample NEF in tests/data/ to run this suite."
)
def test_preprocess_file_end_to_end(tmp_thumb_root: pathlib.Path) -> None:
    """
    preprocess_file should return a populated PreprocessResult and
    reuse cached thumbnails on subsequent calls.
    """
    result1 = preprocess.preprocess_file(
        SAMPLE_NEF, thumb_root=tmp_thumb_root, size=512
    )

    # Contract checks
    assert result1.thumb_path.exists()
    assert result1.rgb is not None and result1.rgb.shape[2] == 3
    assert "DateTimeOriginal" in result1.exif

    # Call again – must hit cache, not rewrite the file (mtime unchanged)
    mtime_before = result1.thumb_path.stat().st_mtime
    result2 = preprocess.preprocess_file(
        SAMPLE_NEF, thumb_root=tmp_thumb_root, size=512
    )
    mtime_after = result2.thumb_path.stat().st_mtime

    assert mtime_before == mtime_after, "Thumbnail was regenerated instead of reused"

    # Clean up thumbnails generated during the test
    shutil.rmtree(tmp_thumb_root)
