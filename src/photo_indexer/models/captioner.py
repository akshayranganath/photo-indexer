"""
photo_indexer.models.captioner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calls an *Ollama* vision-language model (default **llava-next:latest**) and
returns a single-sentence description of the image.

Key points
----------
* Sends the image as *base-64 PNG* to the Ollama REST endpoint
  (`POST /api/generate`).
* Uses a configurable prompt template::
      "Describe the scene in one sentence. Mention place if obvious."
* Caches the HTTP session + model name so you pay init cost only once.
* Accepts PIL.Image, NumPy ndarray or torch.Tensor just like the other heads.
* Returns ``{'caption': str}`` suitable for the fusion layer.

Environment variables
---------------------
* ``OLLAMA_HOST`` – override host:port (default ``http://localhost:11434``).
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any

import numpy as np
import requests
import torch
from PIL import Image

try:
    from photo_indexer.utils.logging import get_logger
except ImportError:  # pragma: no cover
    import logging

    def get_logger(name=None, **kw):  # type: ignore
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)


log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Captioner                                                                   #
# --------------------------------------------------------------------------- #
class Captioner:
    """
    Minimal LLaVA/LLaVA-NeXT client that hits the Ollama REST API.
    """

    _DEFAULT_PROMPT = (
        "Describe the scene in one concise sentence. "
        "If the location is obvious (e.g. beach, glacier, city street) mention it."
    )

    def __init__(
        self,
        #model: str = "llava-next:latest",
        model: str = "llama3.2-vision:latest",
        *,
        api_host: str | None = None,
        prompt_template: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 60,
    ) -> None:
        """
        Parameters
        ----------
        model:
            Name/tag you `ollama pull`’d (e.g. ``llava-next:Q4_K_M``).
        api_host:
            Override for the Ollama base URL (default:
            ``$OLLAMA_HOST`` or ``http://localhost:11434``).
        prompt_template:
            Instruction placed *before* the image; use ``{image}`` as placeholder
            if you need it (rare). Defaults to a built-in one-liner.
        temperature / max_tokens:
            Standard text-generation params forwarded to Ollama.
        """
        self.model = model
        self.api_host = api_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.prompt = prompt_template or self._DEFAULT_PROMPT
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._session = requests.Session()

        # Quick sanity check: ping model list once
        try:
            self._session.get(f"{self.api_host}/api/tags", timeout=2).raise_for_status()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Could not connect to Ollama at {self.api_host}. Is the daemon running?"
            ) from exc

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def __call__(self, image: Any) -> dict[str, str]:
        """
        Run LLaVA on *image* and return ``{'caption': str}``.
        """
        img_b64 = self._to_base64_png(image)
        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "images": [img_b64],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": 0,
                "max_tokens": self.max_tokens,
            },
        }

        log.debug("POST %s/api/generate  [model=%s]", self.api_host, self.model)
        resp = self._session.post(f"{self.api_host}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        caption = data.get("response", "").strip()

        if not caption:
            log.warning("Empty caption for image!")
        return {"caption": caption}

    # --------------------------------------------------------------------- #
    # Internals                                                             #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _to_base64_png(x: Any) -> str:
        """
        Convert supported image types to *base64-encoded* PNG string
        (without the data-URI prefix – Ollama wants plain b64).
        """
        if isinstance(x, Image.Image):
            img = x.convert("RGB")
        elif isinstance(x, np.ndarray):
            if x.ndim == 2:  # gray → RGB
                x = np.stack([x] * 3, axis=-1)
            img = Image.fromarray(x.astype("uint8"), mode="RGB")
        elif isinstance(x, torch.Tensor):
            if x.max() <= 1.0:
                x = x * 255
            x = x.clip(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            img = Image.fromarray(x, mode="RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(x)}")

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)  # PNG compresses nicely
        return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# Manual test (python -m photo_indexer.models.captioner <image>)              #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import sys
    from pathlib import Path

    if len(sys.argv) != 2:
        print("Usage: python -m photo_indexer.models.captioner <image>")
        sys.exit(1)

    img_path = Path(sys.argv[1]).expanduser()
    cap = Captioner()
    result = cap(Image.open(img_path))
    print(result)
