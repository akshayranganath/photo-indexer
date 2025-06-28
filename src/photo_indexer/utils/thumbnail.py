"""
photo_indexer.utils.thumbnail
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Small helpers around *Pillow* for creating and caching JPEG thumbnails.

Public API
----------

``make_thumb_path(raw_path, photo_root, thumb_root) -> Path``  
    Mirror the RAW-file folder hierarchy under *thumb_root* and return
    the final ``*.jpg`` path (directories are created on demand).

``write_thumbnail(rgb, thumb_path, size=512, quality=90) -> None``  
    Resize the input RGB NumPy array so that its **longest edge equals
    *size*** and encode it as JPEG.  Skips work if the file already
    exists and is newer than the caller's source data.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------#
# helpers                                                                     #
# ---------------------------------------------------------------------------#
def _scaled_dims(w: int, h: int, target: int) -> Tuple[int, int]:
    """Return new (w, h) so that *max(w, h) == target* and aspect ratio stays."""
    if w >= h:
        return target, max(1, round(h * target / w))
    return max(1, round(w * target / h)), target


# ---------------------------------------------------------------------------#
# public API                                                                  #
# ---------------------------------------------------------------------------#
def make_thumb_path(
    raw_path: Path,
    photo_root: Path,
    thumb_root: Path,
    ext: str = ".jpg",
) -> Path:
    """
    Compute the thumbnail path by preserving the sub-folder hierarchy
    relative to *photo_root* and swapping the extension to *ext* (default
    ``.jpg``).

        RAW  :  /photos/2024/IMG_0001.NEF
        THUMB:  <thumb_root>/2024/IMG_0001.jpg
    """
    rel = raw_path.relative_to(photo_root).with_suffix(ext)
    thumb_path = thumb_root / rel
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    return thumb_path


def write_thumbnail(
    rgb: np.ndarray,
    thumb_path: Path,
    *,
    size: int = 512,
    quality: int = 90,
    src_mtime: float | None = None,
) -> None:
    """
    Resize *rgb* (uint8 H×W×3) so that its longest edge becomes *size*
    pixels and save as high-quality JPEG.

    * If *thumb_path* already exists **and** is newer than *src_mtime*
      (if given) the function returns immediately.
    * Directory hierarchy is created automatically.
    """
    # caching ----------------------------------------------------------------
    if thumb_path.exists() and src_mtime is not None:
        if thumb_path.stat().st_mtime >= src_mtime:
            return  # up-to-date

    h, w, _ = rgb.shape
    new_w, new_h = _scaled_dims(w, h, size)

    img = Image.fromarray(rgb, mode="RGB").resize((new_w, new_h), Image.LANCZOS)

    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(thumb_path, "JPEG", quality=quality, optimize=True, progressive=True)

    # keep original timestamp handy for cache validators
    if src_mtime is not None:
        os_utime = getattr(Path, "touch", None)  # windows shim
        if os_utime:
            thumb_path.touch(times=(src_mtime, time.time()))
