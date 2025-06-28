"""
photo_indexer.utils.logging
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A thin wrapper around Python’s ``logging`` that gives you:

* 🌈 Rich-styled, colourised console output (falls back to plain text if
  `rich` isn’t installed).
* 📄 Optional daily-rotating file handler that keeps a week of history.
* 🔒 Singleton config—subsequent ``get_logger()`` calls just fetch loggers
  without re-initialising the root handlers.

Example
-------
>>> from photo_indexer.utils.logging import get_logger
>>> log = get_logger(__name__, level=logging.DEBUG, log_file="logs/run.log")
>>> log.info("Indexer started")
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

try:
    from rich.logging import RichHandler
except ImportError:  # pragma: no cover
    RichHandler = None  # type: ignore

# --------------------------------------------------------------------------- #
# Internal state                                                              #
# --------------------------------------------------------------------------- #
_LOG_CONFIGURED = False
_DEFAULT_LEVEL = logging.INFO


def _configure_root(level: int = _DEFAULT_LEVEL, log_file: Optional[Path] = None) -> None:
    """
    One-time initialisation of the *root* logger.

    Console logs:
        • RichHandler if available (pretty colours, tracebacks)
        • Otherwise classic StreamHandler with timestamp.

    File logs (optional):
        • Daily rotation at midnight
        • 7 backups kept (≈ one week of history)
    """
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return

    handlers: list[logging.Handler] = []

    # -------- Console handler ------------------------------------------------
    if RichHandler is not None:
        handlers.append(
            RichHandler(
                level=level,
                show_time=True,
                show_level=True,
                show_path=False,
                rich_tracebacks=True,
            )
        )
    else:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        handlers.append(stream_handler)

    # -------- File handler ---------------------------------------------------
    if log_file:
        log_file = Path(log_file).expanduser().resolve()
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=str(log_file),
            when="midnight",
            backupCount=7,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | "
                "%(name)s (%(funcName)s:%(lineno)d): %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(file_handler)

    # -------- Activate root config ------------------------------------------
    logging.basicConfig(level=level, handlers=handlers)
    _LOG_CONFIGURED = True


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def get_logger(
    name: str | None = None,
    *,
    level: int = _DEFAULT_LEVEL,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """
    Return a configured ``logging.Logger``.

    Parameters
    ----------
    name:
        Usually ``__name__`` of the caller. ``None`` returns the root logger.
    level:
        Lowest level that will appear in both console and file logs
        (default: ``logging.INFO``).
    log_file:
        Path where a rotating log file should be written. If *None*, no file
        logging is set up.

    Examples
    --------
    >>> log = get_logger(__name__)
    >>> log.debug("Verbose details")
    """
    _configure_root(level=level, log_file=Path(log_file) if log_file else None)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
