"""
photo_indexer.config
~~~~~~~~~~~~~~~~~~~~

Typed configuration object + tiny loader that merges defaults with values
from an optional **YAML** file.

Typical usage
-------------
>>> from photo_indexer.config import load_config
>>> cfg = load_config()                          # ~/.config/photo_indexer/config.yaml
>>> print(cfg.workers, cfg.caption_model)

You can also point it at any file:
>>> cfg = load_config("/path/to/my_settings.yaml")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, Field, validator

__all__ = ["IndexerSettings", "load_config"]


# --------------------------------------------------------------------------- #
# Settings dataclass (immutable)                                              #
# --------------------------------------------------------------------------- #
class IndexerSettings(BaseModel):
    # --- General --------------------------------------------------------- #
    workers: int = Field(8, gt=0, description="Max concurrent worker threads")
    db_path: Path = Field(
        Path("~/photo_index.sqlite").expanduser(),
        description="SQLite (or DuckDB) file where rows are stored",
    )

    # --- Pre-processing -------------------------------------------------- #
    thumbnail_size: int = Field(512, gt=64, description="Pixels on shortest side")

    # --- Models ---------------------------------------------------------- #
    scene_model: str = Field(
        "mobilenetv2-places365", description="Weights tag or path for scene net"
    )
    people_model: str = Field(
        "yolov10s.pt", description="Weights tag or path for YOLO person detector"
    )
    caption_model: str = Field(
        "llama3.2-vision:latest", description="Ollama model tag for captioning"
    )

    # --- External services ---------------------------------------------- #
    ollama_host: str = Field(
        "http://localhost:11434", description="Base URL of the Ollama daemon"
    )

    class Config:
        frozen = True  # make instance hashable / read-only
        allow_mutation = False
        extra = "ignore"

    # Validate paths so that they always expand user (~)
    @validator("db_path", pre=True)
    def _expand_path(cls, v: Path | str) -> Path:  # noqa: N805
        return Path(v).expanduser()


# --------------------------------------------------------------------------- #
# Loader helper                                                               #
# --------------------------------------------------------------------------- #
def load_config(path: str | os.PathLike | None = None) -> IndexerSettings:
    """
    Load settings from *path* (YAML). Missing keys fall back to defaults.

    If *path* is ``None`` and ``~/.config/photo_indexer/config.yaml`` exists,
    that file is loaded automatically. Otherwise, purely default settings
    are returned.

    Raises
    ------
    FileNotFoundError
        When *path* is given explicitly but does not exist.
    yaml.YAMLError
        When the file cannot be parsed.
    """
    if path is None:
        path = Path("~/.config/photo_indexer/config.yaml").expanduser()
        if not path.exists():
            return IndexerSettings()

    yaml_path = Path(path).expanduser()
    if not yaml_path.exists():
        raise FileNotFoundError(yaml_path)

    with yaml_path.open("r") as fh:
        data: Mapping[str, Any] = yaml.safe_load(fh) or {}

    return IndexerSettings(**data)
