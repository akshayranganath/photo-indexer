"""
photo_indexer.cli
~~~~~~~~~~~~~~~~~

Command-line interface for *Photo-Indexer*.

The CLI is a thin wrapper that

1. Boots the global logging system.
2. Parses user options (via **Click**).
3. Delegates heavy lifting to :pymod:`photo_indexer.workers`.

You get a single sub-command:

    $ pi index /path/to/RAWs  [OPTIONS]

which walks every *.NEF*, runs the full vision pipeline, and stores the
results in the chosen database backend.

The entry-point name **`pi`** is registered in *pyproject.toml* so it
becomes available after

    $ pip install -e .

or

    $ poetry install
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import click

from photo_indexer.utils.logging import get_logger #,setup
from photo_indexer.config import IndexerSettings

# -- lazy import to avoid torch startup when showing --help -------------------
def _lazy_worker_import():
    #from photo_indexer.workers import run_index  # local import to defer heavy deps    
    from photo_indexer.workers import index_folder

    return index_folder


_log = get_logger(__name__)


# ---------------------------------------------------------------------------#
# Click helpers                                                               #
# ---------------------------------------------------------------------------#
class _PathExists(click.Path):
    """Click Path subtype that enforces *exists=True* by default."""

    def __init__(self, **kwargs):
        super().__init__(exists=True, file_okay=True, dir_okay=True, readable=True, **kwargs)


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def cli() -> None:  # pragma: no cover
    """Photo-Indexer command-line tool."""
    # The group does nothing by itself; sub-commands below do the work.


@cli.command("index", help="Index all .NEF photos under PHOTO_ROOT.")
@click.argument("photo_root", type=_PathExists(path_type=Path))
@click.option(
    "--workers",
    "-w",
    type=int,
    metavar="N",
    default=os.cpu_count(),
    show_default="CPU core count",
    help="Concurrent worker threads.",
)
@click.option(
    "--db",
    type=click.Choice(["sqlite", "duckdb"], case_sensitive=False),
    default="sqlite",
    show_default=True,
    help="Storage backend.",
)
@click.option(
    "--thumb-size",
    type=int,
    default=512,
    show_default=True,
    help="Longest edge of cached JPEG thumbnails (pixels).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable DEBUG-level logging.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Run pipeline but skip final DB insert (for timing tests).",
)
def cmd_index(
    photo_root: Path,
    workers: int,
    db: str,
    thumb_size: int,
    verbose: bool,
    dry_run: bool,
) -> None:
    """
    Walk PHOTO_ROOT recursively, process every *.NEF* and write results to
    a local database (default *data/db/photo_index.sqlite*).
    """
    #setup(verbose=verbose)

    _log.info("Photo-Indexer starting (root=%s, workers=%d)", photo_root, workers)

    index_folder = _lazy_worker_import()

    try:
        # Map database backend to file path
        db_file = f"data/db/photo_index.{db.lower()}"
        
        settings = IndexerSettings(
            workers=workers,
            db_path=Path(db_file),
            thumbnail_size=thumb_size,
        )
        index_folder(photo_root, settings=settings, dry_run=dry_run)
    except KeyboardInterrupt:
        _log.warning("Interrupted by user – exiting.")
        sys.exit(130)
    except Exception as exc:  # pylint: disable=broad-except
        _log.exception("Fatal error: %s", exc)
        sys.exit(1)

    _log.info("Done – bye.")


# ---------------------------------------------------------------------------#
# Stand-alone invocation (python -m photo_indexer.cli)                        #
# ---------------------------------------------------------------------------#
def main() -> None:  # pragma: no cover
    """Module-level entry-point so `python -m photo_indexer.cli …` works."""
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
