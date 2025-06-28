"""
photo_indexer.models.scene_classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Thin wrapper around the AlexNet-Places365 network.

* First call transparently downloads the ~233 MB weight file and the
  label/IO text files into  ``~/.cache/photo_indexer/``.
* Accepts either a `PIL.Image`, a `numpy.ndarray` (H × W × 3, RGB),
  or a `torch.Tensor` (C × H × W, [0 … 1]).
* Returns a **dict**::

    {'label': 'mountain', 'indoor': False}

If you want the top-k labels or raw logits, feel free to extend
``__call__``—the model and category list are exposed as attributes.
"""

from __future__ import annotations

import hashlib
import pathlib
import urllib.request
from typing import Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import alexnet
from PIL import Image
import numpy as np

# --------------------------------------------------------------------------- #
# Constants & helper URLs                                                     #
# --------------------------------------------------------------------------- #
_CACHE = pathlib.Path.home() / ".cache" / "photo_indexer"
_CACHE.mkdir(parents=True, exist_ok=True)

# Using PyTorch weights instead of Caffe model
_WEIGHTS_URL = (    
    "http://places2.csail.mit.edu/models_places365/"
    "alexnet_places365.pth.tar"    
)

_CATEGORIES_URL = (
    "https://raw.githubusercontent.com/csailvision/places365/master"
    "/categories_places365.txt"
)
_IO_URL = (
    "https://raw.githubusercontent.com/csailvision/places365/master"
    "/IO_places365.txt"
)


def _download(url: str, dest: pathlib.Path, sha256: str | None = None) -> None:
    """Download *url* to *dest* (if not already). Optionally verify SHA-256."""
    if dest.exists():
        return
    print(f"[SceneClassifier] → downloading {dest.name} …")
    urllib.request.urlretrieve(url, dest)  # nosec B310
    if sha256:
        h = hashlib.sha256(dest.read_bytes()).hexdigest()
        if h != sha256:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"SHA256 mismatch for {dest}: {h} vs {sha256}")


def _load_txt_lines(path: pathlib.Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


# --------------------------------------------------------------------------- #
# SceneClassifier                                                              #
# --------------------------------------------------------------------------- #
class SceneClassifier:
    """Lazy-loads weights and provides a simple ``__call__`` interface."""

    def __init__(self, device: str | torch.device | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # ---------- Make sure files are on disk --------------------------------
        weights_path = _CACHE / "alexnet_places365.pth.tar"
        cats_path = _CACHE / "categories_places365.txt"
        io_path = _CACHE / "IO_places365.txt"

        _download(_WEIGHTS_URL, weights_path)
        _download(_CATEGORIES_URL, cats_path)
        _download(_IO_URL, io_path)

        # ---------- Categories & indoor/outdoor list ---------------------------
        # categories file: "abbey n03425413" – we only need the category name
        self.categories: list[str] = [ln.split(" ")[0] for ln in _load_txt_lines(cats_path)]
        # Parse IO file: each line contains "category_name index" where index is 0 (indoor) or 1 (outdoor)
        self.io_list: list[int] = [int(line.split(" ")[1]) for line in _load_txt_lines(io_path)]

        # ---------- Model ------------------------------------------------------
        self.model = alexnet(pretrained=False)  # Don't load ImageNet weights
        # replace the classifier (last layer) – Places365 has 365 classes
        self.model.classifier[6] = torch.nn.Linear(4096, 365)
        
        # Load Places365 pre-trained weights (PyTorch format)
        try:
            checkpoint = torch.load(weights_path, map_location="cpu")
            # Handle different possible state dict formats
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            # Strip "module." prefix if present (weights trained via DataParallel)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Could not load pre-trained weights from {weights_path}")
            print(f"Error: {e}")
            print("Using randomly initialized AlexNet instead.")
        
        self.model.eval().to(self.device)

        # ---------- Pre-processing pipeline ------------------------------------
        self.preproc = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def __call__(self, image) -> dict[str, str | bool]:
        """Classify a single image and return *coarse* label + indoor flag."""
        tensor = self._to_tensor(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)

        idx: int = int(probs.argmax())
        fine_label = self.categories[idx]              # e.g. "snowfield"
        coarse = fine_label.split("/")[0].split("_")[0]  # → "snowfield"

        indoor = self.io_list[idx] == 0
        return {"label": coarse, "indoor": indoor}

    # --------------------------------------------------------------------- #
    # Internals                                                             #
    # --------------------------------------------------------------------- #
    def _to_tensor(self, img) -> torch.Tensor:
        """Convert PIL / ndarray / tensor input to normalised torch.Tensor."""
        if isinstance(img, torch.Tensor):
            if img.max() > 1.1:  # assume [0-255] uint8 tensor
                img = img / 255.0
            if img.ndim == 3 and img.shape[0] == 3:
                return self.preproc.transforms[-1](img)  # only normalise
            # else fall through to pillow path for resizing etc.

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype("uint8"), mode="RGB")
        elif not isinstance(img, Image.Image):
            raise TypeError(f"Unsupported image type: {type(img)}")

        return self.preproc(img)


# --------------------------------------------------------------------------- #
# Quick manual test (python -m photo_indexer.models.scene_classifier <file>)  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import sys
    from pathlib import Path
    from PIL import Image

    if len(sys.argv) != 2:
        print("Usage: python -m photo_indexer.models.scene_classifier <image>")
        sys.exit(1)

    img_path = Path(sys.argv[1]).expanduser()
    sc = SceneClassifier()
    out = sc(Image.open(img_path))
    print(out)
