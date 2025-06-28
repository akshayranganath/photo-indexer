"""
photo_indexer.workers
~~~~~~~~~~~~~~~~~~~~~

* Recursively walks a folder for **.NEF / .nef** RAW files.
* Spawns a `ThreadPoolExecutor` (`settings.workers` threads).
* For every file:

      NEF  ─▶  thumbnail (rawpy) ─▶  vision heads
                                   ├── scene classifier
                                   ├── people detector
                                   └── captioner
           ─▶  metadata fusion  ─▶  row dict
           ─▶  SQLite insert  (or STDOUT in --dry-run mode)

Nothing here is GPU-bound, so a thread pool is simpler (and usually faster)
than an asyncio solution.

The module intentionally depends **only** on public package APIs (`get_model`,
`IndexerSettings`) plus a couple of common libraries; heavy details live in
the specialised sub-modules.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:
    import rawpy
except ImportError:  # pragma: no cover
    rawpy = None  # type: ignore

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback – no progress bar
    def tqdm(x: Iterable, **_):  # type: ignore
        return x

from photo_indexer import get_logger, get_model
from photo_indexer.models import get_model_with_settings
from photo_indexer.config import IndexerSettings

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Public entry-point                                                          #
# --------------------------------------------------------------------------- #
def index_folder(
    root: os.PathLike | str,
    *,
    settings: IndexerSettings,
    dry_run: bool = False,
) -> None:
    """
    Crawl *root* and index every **.NEF** file found.

    Parameters
    ----------
    root:
        Directory to traverse (recursively).
    settings:
        Immutable runtime options (`IndexerSettings`).
    dry_run:
        When *True* rows are **printed to STDOUT** instead of being written
        to the SQLite database.
    """
    root = Path(root).expanduser()
    if not root.is_dir():
        log.error("Path %s is not a directory", root)
        sys.exit(1)

    files: List[Path] = sorted(
        p for p in root.rglob("*") if p.suffix.lower() == ".nef"
    )
    if not files:
        log.warning("No .NEF files found under %s – nothing to do.", root)
        return

    log.info("Found %d RAW files – spinning up %d workers", len(files), settings.workers)

    # SQLite initialisation (noop when dry-run)
    if not dry_run:
        _ensure_db(settings.db_path)

    # Thread pool fan-out
    with ThreadPoolExecutor(max_workers=settings.workers) as pool:
        fut_map = {pool.submit(_process_one, path, settings): path for path in files}

        for fut in tqdm(as_completed(fut_map), total=len(fut_map), unit="img"):
            path = fut_map[fut]
            try:
                row = fut.result()
            except Exception as exc:  # pragma: no cover
                log.exception("❌  %s  (%s)", path.name, exc)
                continue

            if dry_run:
                print(json.dumps(row, ensure_ascii=False))
            else:
                _insert_row(settings.db_path, row)


# --------------------------------------------------------------------------- #
# Worker – one image                                                          #
# --------------------------------------------------------------------------- #
def _process_one(path: Path, settings: IndexerSettings) -> Dict[str, Any]:
    """
    Return a *dict* containing the fused predictions + EXIF info
    for a single image.
    """
    img, exif_dt = _read_nef_thumbnail(path, thumb_px=settings.thumbnail_size)

    # ---- Vision heads ----------------------------------------------------
    scene = get_model("scene")(img)                              # {'label', 'indoor'}
    people = get_model("people")(img)                            # {'people', 'count'}
    caption = get_model_with_settings("caption", settings)(img)  # {'caption'}

    # ---- Business logic fusion ------------------------------------------
    indoor = scene["indoor"]
    scene_name = "indoor" if indoor else "outdoor"
    location = "home" if indoor else scene["label"]

    # Date/time fall back: use file mtime when EXIF missing
    if exif_dt is None:
        exif_dt = datetime.fromtimestamp(path.stat().st_mtime)

    row = {
        "file": str(path),
        "scene": scene_name,
        "location": location,
        "people": bool(people["people"]),
        "count": int(people["count"]),
        "date": exif_dt.strftime("%Y-%m-%d"),
        "time": exif_dt.strftime("%H:%M"),
        "description": caption["caption"],
    }
    return row


# --------------------------------------------------------------------------- #
# I/O helpers                                                                 #
# --------------------------------------------------------------------------- #
def _read_nef_thumbnail(path: Path, *, thumb_px: int = 512) -> tuple[Image.Image, datetime | None]:
    """
    Load NEF → small RGB `PIL.Image` + best-effort EXIF datetime.

    Falls back to file *mtime* when EXIF is absent or unreadable.
    """
    if rawpy is None:  # pragma: no cover
        raise RuntimeError(
            "rawpy is not installed – install the full requirements "
            "to read Nikon .NEF files."
        )

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            output_bps=8,
            no_auto_bright=True,
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.sRGB,
        )
        # Simple nearest resize – speed over quality (thumbnail only)
        img = Image.fromarray(rgb)
        img.thumbnail((thumb_px, thumb_px), Image.Resampling.BILINEAR)

        try:
            exif_str: bytes | None = raw.metadata.datetime  # e.g. b"2024:06:12 18:34:55"
            exif_dt = (
                datetime.strptime(exif_str.decode(), "%Y:%m:%d %H:%M:%S")
                if exif_str
                else None
            )
        except Exception:  # pragma: no cover
            exif_dt = None

    return img, exif_dt


def _ensure_db(path: Path) -> None:
    """
    Create the *photos* table if the DB file is new.
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS photos (
                file       TEXT PRIMARY KEY,
                scene      TEXT,
                location   TEXT,
                people     INTEGER,
                count      INTEGER,
                date       TEXT,
                time       TEXT,
                description TEXT
            )
            """
        )
        conn.commit()


def _insert_row(db_path: Path, row: Dict[str, Any]) -> None:
    """
    Insert / replace a row in the *photos* table.
    """
    with sqlite3.connect(db_path, timeout=30, isolation_level=None) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO photos
            (file, scene, location, people, count, date, time, description)
            VALUES (:file, :scene, :location, :people, :count,
                    :date, :time, :description)
            """,
            row,
        )
