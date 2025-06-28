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


@lru_cache(maxsize=None)
def _load_caption() -> VisionModel:
    from .captioner import Captioner
    return Captioner()


_LOADERS: dict[ModelName, callable[[], VisionModel]] = {
    "scene": _load_scene,
    "people": _load_people,
    "caption": _load_caption,
}

__all__ = ["get_model", "VisionModel"]


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def get_model(name: ModelName) -> VisionModel:
    """
    Return a singleton instance of the requested model.

    Parameters
    ----------
    name:
        One of ``"scene"``, ``"people"``, ``"caption"``.

    Raises
    ------
    ValueError
        If *name* is not a recognised key.
    """
    try:
        return _LOADERS[name]()
    except KeyError as exc:  # pragma: no cover
        raise ValueError(
            f"Unknown model '{name}'. Valid options: {', '.join(_LOADERS)}"
        ) from exc
