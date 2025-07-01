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
        Path("data/db/photo_index.sqlite").expanduser(),
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
    
    # --- Captioning provider settings ----------------------------------- #
    caption_provider: str = Field(
        "ollama", description="Captioning provider: 'ollama' or 'openai'"
    )
    caption_model: str = Field(
        "llama3.2-vision:latest", description="Model name for captioning"
    )
    caption_prompt: str = Field(
        "Describe the scene in one concise sentence. If the location is obvious (e.g. beach, glacier, city street) mention it.",
        description="Custom prompt template for captioning"
    )
    caption_temperature: float = Field(
        0.0, ge=0.0, le=2.0, description="Temperature for caption generation"
    )
    caption_max_tokens: int = Field(
        60, gt=0, description="Maximum tokens for caption generation"
    )

    # --- External services ---------------------------------------------- #
    ollama_host: str = Field(
        "http://localhost:11434", description="Base URL of the Ollama daemon"
    )
    openai_api_key: str | None = Field(
        None, description="OpenAI API key (also from OPENAI_API_KEY env var)"
    )
    openai_base_url: str = Field(
        "https://api.openai.com/v1", description="OpenAI API base URL"
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
