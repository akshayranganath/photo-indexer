"""
photo_indexer.cli
~~~~~~~~~~~~~~~~~

Command-line interface for the Photo Indexer.

Typical invocations
-------------------
# Index a folder with the defaults found in ~/.config/photo_indexer/config.yaml
$ pi index ~/Pictures/DSLR-dump

# Override worker count and DB location on the fly
$ pi index /mnt/photos --workers 12 --db ~/scratch/photos.db

# Show the effective configuration then quit
$ pi config

# Print package version
$ pi version
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from photo_indexer import __version__, get_logger
from photo_indexer.config import IndexerSettings, load_config

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Argument parsing                                                            #
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pi",
        description="Photo Indexer – turn DSLR RAW dumps into a searchable DB",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ----- pi index -------------------------------------------------------
    p_index = sub.add_parser("index", help="Index a folder of RAW/NEF photos")
    p_index.add_argument("folder", type=Path, help="Directory to recurse into")
    p_index.add_argument(
        "-w",
        "--workers",
        type=int,
        help="Override the number of concurrent workers (default from config)",
    )
    p_index.add_argument(
        "--db",
        dest="db_path",
        type=Path,
        help="Override path to SQLite/DuckDB file",
    )
    p_index.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Load settings from an explicit YAML file",
    )
    p_index.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pipeline but *do not* write to the DB; emit JSON rows to stdout",
    )

    # ----- pi config ------------------------------------------------------
    p_cfg = sub.add_parser("config", help="Print the merged configuration")
    p_cfg.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Explicit YAML file to merge with defaults",
    )

    # ----- pi version -----------------------------------------------------
    sub.add_parser("version", help="Show package version and exit")

    return parser


# --------------------------------------------------------------------------- #
# Sub-command handlers                                                        #
# --------------------------------------------------------------------------- #
def _run_index(args: argparse.Namespace) -> None:
    cfg: IndexerSettings = load_config(args.config)

    # Merge CLI overrides
    updates = {}
    if args.workers:
        updates["workers"] = args.workers
    if args.db_path:
        updates["db_path"] = args.db_path
    if updates:
        cfg = cfg.copy(update=updates)  # type: ignore[arg-type]

    log.info(
        "Starting indexing run: folder=%s  workers=%d  db=%s",
        args.folder,
        cfg.workers,
        cfg.db_path,
    )

    try:
        # Lazy import keeps CLI fast when user only wants `pi version`
        from photo_indexer.pipelines.workers import index_folder  # noqa: WPS433

        index_folder(
            root=args.folder,
            settings=cfg,
            dry_run=args.dry_run,
        )
    except ImportError as exc:  # pragma: no cover
        log.error(
            "Pipeline modules are missing or failed to import: %s\n"
            "Make sure you've installed all runtime dependencies.",
            exc,
        )
        sys.exit(1)


def _show_config(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    import pprint

    pprint.pprint(cfg.dict())


# --------------------------------------------------------------------------- #
# Public entry-point                                                          #
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> None:  # noqa: D401
    """Entry-point callable for ``pi`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    match args.command:
        case "index":
            _run_index(args)
        case "config":
            _show_config(args)
        case "version":
            print(__version__)
        case _:  # pragma: no cover
            parser.error(f"Unknown command {args.command!r}")


if __name__ == "__main__":  # pragma: no cover
    main()
