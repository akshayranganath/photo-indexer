"""
tests/test_models.py
~~~~~~~~~~~~~~~~~~~~

Smoke-tests for the three vision adapters in *src/photo_indexer/models/*

The goal is **not** to verify model accuracy (that is covered upstream)
but to ensure that:

* Every adapter can be instantiated without crashing.
* The `predict()` call on a tiny in-memory RGB image returns values
  matching the pipeline contract.

To keep CI lightweight we feed a **64×64 random RGB array**; the content
doesn’t matter—we only assert on *types* and *value ranges*.

If the required model weights are missing on the local machine the
corresponding test is skipped with a helpful message instead of failing.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# build a tiny dummy image once for all tests
DUMMY_IMG = (np.random.rand(64, 64, 3) * 255).astype("uint8")

# helper: skip if weight file does not exist -------------------------------
def _skip_if_missing(path: Path, reason: str) -> None:
    if not path.exists():
        pytest.skip(f"{reason} – skipping test", allow_module_level=True)


# ---------------------------------------------------------------------------
# Scene Classifier – ResNet-18 Places365
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_path", [
    Path("data/models/resnet18_places365.pth.tar"),        # default location
    Path(os.getenv("PLACES_WEIGHTS", ""))                  # or env override
])
def test_scene_classifier_predict(model_path: Path) -> None:
    """SceneClassifier.predict should produce ('indoor'|'outdoor', location)."""
    if model_path:
        _skip_if_missing(model_path, "ResNet-18 Places365 weights absent")

    from photo_indexer.models.scene_classifier import SceneClassifier

    classifier = SceneClassifier(weight_path=model_path or None)
    scene, location = classifier.predict(DUMMY_IMG)

    assert scene in {"indoor", "outdoor"}, "scene must be 'indoor' or 'outdoor'"
    assert isinstance(location, str) and location, "location must be non-empty str"


# ---------------------------------------------------------------------------
# People Detector – YOLOv8-n
# ---------------------------------------------------------------------------
def test_people_detector_predict() -> None:
    """PeopleDetector.predict returns non-negative person count."""
    from photo_indexer.models.people_detector import PeopleDetector

    detector = PeopleDetector()   # weights auto-download on first call
    count = detector.predict(DUMMY_IMG)

    assert isinstance(count, int) and count >= 0


# ---------------------------------------------------------------------------
# Captioner – LLaVA via Ollama REST
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not os.getenv("OLLAMA_HOST", "http://localhost:11434").startswith("http"),
    reason="OLLAMA_HOST env var mis-configured",
)
def test_captioner_predict() -> None:
    """
    Captioner.predict returns a short non-empty string.
    Requires Ollama server to be running and the model to be pulled:
        ollama pull llava:7b-q4_k_m
    """
    from photo_indexer.models.captioner import Captioner

    cap = Captioner(model="llava:7b-q4_k_m")
    text = cap.predict(DUMMY_IMG)

    assert isinstance(text, str) and len(text) > 10, "caption string too short"
