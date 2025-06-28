"""
photo_indexer.models
~~~~~~~~~~~~~~~~~~~~

Central point for loading the three vision heads used by the pipeline:

* **scene**   → MobileNetV2-Places365 wrapper
* **people**  → YOLOv10-s person detector / counter
* **caption** → Llama3.2-Vision captioner (Ollama client)

The public helper :func:`get_model` hands back a *singleton* of the model you
ask for, so repeated calls are cheap.

Example
-------
>>> from photo_indexer.models import get_model
>>> scene_cls = get_model("scene")
>>> preds = scene_cls(image)        # {'label': 'mountain', 'indoor': False}
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Protocol, runtime_checkable

# --------------------------------------------------------------------------- #
# Types                                                                       #
# --------------------------------------------------------------------------- #
ModelName = Literal["scene", "people", "caption"]


@runtime_checkable
class VisionModel(Protocol):
    """Minimal interface all vision heads expose."""

    def __call__(self, image, *args, **kwargs):
        ...


# --------------------------------------------------------------------------- #
# Internal lazy-loaders                                                       #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _load_scene() -> VisionModel:
    from .scene_classifier import SceneClassifier  # Local import = lazy
    return SceneClassifier()


@lru_cache(maxsize=None)
def _load_people() -> VisionModel:
    from .people_detector import PeopleDetector
    return PeopleDetector()


def _load_caption(settings=None) -> VisionModel:
    from .captioner import Captioner
    from photo_indexer.config import IndexerSettings
    
    if settings is None:
        # Use default settings if none provided
        settings = IndexerSettings()
    
    return Captioner(
        provider=settings.caption_provider,
        model=settings.caption_model,
        ollama_host=settings.ollama_host,
        openai_api_key=settings.openai_api_key,
        openai_base_url=settings.openai_base_url,
        prompt_template=settings.caption_prompt,
        temperature=settings.caption_temperature,
        max_tokens=settings.caption_max_tokens,
    )


_LOADERS: dict[ModelName, callable[[], VisionModel]] = {
    "scene": _load_scene,
    "people": _load_people,
}

# Cache for settings-based caption model
_caption_cache: dict[str, VisionModel] = {}

__all__ = ["get_model", "get_model_with_settings", "VisionModel"]


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def get_model(name: ModelName) -> VisionModel:
    """
    Return a singleton instance of the requested model using default settings.

    Parameters
    ----------
    name:
        One of ``"scene"``, ``"people"``, ``"caption"``.

    Raises
    ------
    ValueError
        If *name* is not a recognised key.
    
    Note
    ----
    For caption models, this uses default settings. Use `get_model_with_settings`
    for custom configuration.
    """
    if name == "caption":
        return get_model_with_settings(name, None)
    
    try:
        return _LOADERS[name]()
    except KeyError as exc:  # pragma: no cover
        raise ValueError(
            f"Unknown model '{name}'. Valid options: {', '.join(_LOADERS)} or caption"
        ) from exc


def get_model_with_settings(name: ModelName, settings) -> VisionModel:
    """
    Return a model instance configured with the provided settings.

    Parameters
    ----------
    name:
        One of ``"scene"``, ``"people"``, ``"caption"``.
    settings:
        IndexerSettings instance with model configuration.

    Raises
    ------
    ValueError
        If *name* is not a recognised key.
    """
    if name == "caption":
        # Create a cache key based on settings to enable caching
        if settings is None:
            cache_key = "default"
        else:
            cache_key = f"{settings.caption_provider}:{settings.caption_model}:{settings.ollama_host}:{settings.openai_base_url}"
        
        if cache_key not in _caption_cache:
            _caption_cache[cache_key] = _load_caption(settings)
        return _caption_cache[cache_key]
    
    # For other models, settings don't matter yet
    return get_model(name)
