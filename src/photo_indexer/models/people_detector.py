"""
photo_indexer.models.people_detector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detects human figures in an image using Ultralytics **YOLO v10-s**.

* Loads the model lazily; first run downloads `yolov10s.pt` into the
  Ultralytics cache (usually `~/.cache/ultralytics`).
* Accepts a `PIL.Image`, NumPy `ndarray` (H × W × 3, RGB) or
  PyTorch `Tensor` (C × H × W, [0 … 1]).
* Returns a **dict**::

      {"people": True, "count": 3}

If you need the raw bounding boxes, you can poke
`result.boxes.xyxy` after the call—this wrapper keeps them in
`self._last_boxes` for convenience.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

try:
    from photo_indexer.utils.logging import get_logger
except ImportError:  # unit-tests can monkey-patch
    import logging

    def get_logger(name=None, **kw):  # type: ignore
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)


log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# PeopleDetector                                                              #
# --------------------------------------------------------------------------- #
class PeopleDetector:
    """
    Thin wrapper around YOLOv10-s that yields only “person” detections.
    """

    def __init__(
        self,
        weights: str | Path = "yolov10s.pt",
        device: str | torch.device | None = None,
        conf: float = 0.25,
        iou: float = 0.6,
    ) -> None:
        """
        Parameters
        ----------
        weights:
            Path or Hub tag for the YOLOv10 model (default: ``yolov10s.pt``).
        device:
            `'cuda'`, `'mps'`, `'cpu'` or explicit ``torch.device``.
            Default chooses GPU if available.
        conf:
            Confidence threshold for a box to be kept.
        iou:
            IoU threshold for the NMS-free E-Max decoder (kept for parity with v8).
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        log.debug("Loading YOLO model (%s) on %s", weights, self.device)
        self.model: YOLO = YOLO(str(weights))
        self.model.to(self.device)
        # Limit to class-0 = person
        self.model.overrides["classes"] = [0]
        self.model.overrides["conf"] = conf
        self.model.overrides["iou"] = iou

        # last inference artefacts (useful for debugging / tests)
        self._last_result: Results | None = None
        self._last_boxes: torch.Tensor | None = None

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def __call__(self, image: Any) -> dict[str, int | bool]:
        """
        Run inference on *image* and return::

            {"people": <bool>, "count": <int>}
        """
        img = self._to_img(image)
        # YOLO returns a Results list – we process only idx 0
        self._last_result = self.model(img, verbose=False)[0]
        self._last_boxes = self._last_result.boxes.xyxy  # (N × 4) or empty

        count = int(self._last_boxes.shape[0]) if self._last_boxes is not None else 0
        return {"people": bool(count), "count": count}

    # --------------------------------------------------------------------- #
    # Internals                                                             #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _to_img(x: Any):
        """
        Convert input to a format Ultralytics accepts (np.uint8 RGB or PIL).
        """
        if isinstance(x, Image.Image):
            return x  # already OK

        if isinstance(x, np.ndarray):
            if x.dtype != np.uint8:
                x = x.astype("uint8")
            if x.ndim == 2:  # gray → RGB
                x = np.stack([x] * 3, axis=-1)
            return x

        if isinstance(x, torch.Tensor):
            # Convert to uint8 ndarray to avoid torch → PIL slow path
            if x.max() <= 1.01:
                x = x * 255
            x = x.clip(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            return x

        raise TypeError(f"Unsupported image type: {type(x)}")


# --------------------------------------------------------------------------- #
# Manual test (python -m photo_indexer.models.people_detector <image>)        #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m photo_indexer.models.people_detector <image>")
        sys.exit(1)

    img_path = Path(sys.argv[1]).expanduser()
    det = PeopleDetector()
    out = det(Image.open(img_path))
    print(out)
    # Visual check: det._last_result.show()
