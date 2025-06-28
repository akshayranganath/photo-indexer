"""
photo_indexer
~~~~~~~~~~~~~

Public façade for the Photo Indexer package.

* Exposes **get_model** (lazy singleton loader for the three vision heads)
  and **get_logger** (Rich-enabled helper) at the top level so callers can:

      >>> from photo_indexer import get_model, get_logger
      >>> log = get_logger(__name__)
      >>> scene_net = get_model("scene")

* Provides a defensively-set ``__version__`` (falls back to “0.0.0” when the
  package is run from source).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__: str = _pkg_version(__name__)
except PackageNotFoundError:  # e.g. running from a git checkout
    __version__ = "0.0.0"

# Convenience re-exports
from .models import get_model           # noqa: E402  (circular safe – lazy import)
from .utils.logging import get_logger   # noqa: E402

__all__ = ["get_model", "get_logger", "__version__"]
