"""
photo_indexer.models.captioner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Flexible vision-language captioning client that supports both local (Ollama) and 
remote (OpenAI) providers for generating image descriptions.

Key points
----------
* **Ollama provider**: Supports local models like llama3.2-vision:latest
* **OpenAI provider**: Supports GPT-4 Vision models (gpt-4-vision-preview, gpt-4o, etc.)
* Unified interface regardless of provider
* Accepts PIL.Image, NumPy ndarray or torch.Tensor 
* Returns ``{'caption': str}`` for consistent integration

Environment variables
---------------------
* ``OLLAMA_HOST`` – override Ollama host:port (default ``http://localhost:11434``)
* ``OPENAI_API_KEY`` – required for OpenAI provider
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Literal

import numpy as np
import requests
import torch
from PIL import Image
import json 

try:
    from photo_indexer.utils.logging import get_logger
except ImportError:  # pragma: no cover
    import logging

    def get_logger(name=None, **kw):  # type: ignore
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)


log = get_logger(__name__)

ProviderType = Literal["ollama", "openai"]


# --------------------------------------------------------------------------- #
# Captioner                                                                   #
# --------------------------------------------------------------------------- #
class Captioner:
    """
    Unified captioning client supporting both Ollama (local) and OpenAI (remote) providers.
    """

    _DEFAULT_PROMPT = (
        "Describe the scene in one concise sentence. "
        "If the location is obvious (e.g. beach, glacier, city street) mention it."
    )

    def __init__(
        self,
        provider: ProviderType = "ollama",
        model: str | None = None,
        *,
        # Ollama-specific
        ollama_host: str | None = None,
        # OpenAI-specific  
        openai_api_key: str | None = None,
        openai_base_url: str = "https://api.openai.com/v1",
        # Common parameters
        prompt_template: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 60,
    ) -> None:
        """
        Parameters
        ----------
        provider:
            Choose between "ollama" (local) or "openai" (remote).
        model:
            Model name. Defaults:
            - Ollama: "llama3.2-vision:latest"  
            - OpenAI: "gpt-4o"
        ollama_host:
            Ollama base URL (default: $OLLAMA_HOST or http://localhost:11434).
        openai_api_key:
            OpenAI API key (default: $OPENAI_API_KEY). Required for OpenAI provider.
        openai_base_url:
            OpenAI API base URL for custom endpoints.
        prompt_template:
            Custom prompt template. Defaults to built-in scene description prompt.
        temperature / max_tokens:
            Standard generation parameters.
        """
        self.provider = provider
        self.prompt = prompt_template or self._DEFAULT_PROMPT
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._session = requests.Session()

        if provider == "ollama":
            self.model = model or "llama3.2-vision:latest"
            self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
            self._setup_ollama()
        elif provider == "openai":
            self.model = model or "gpt-4o"
            self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            self.openai_base_url = openai_base_url
            self._setup_openai()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _setup_ollama(self) -> None:
        """Initialize Ollama provider with connectivity check."""
        try:
            resp = self._session.get(f"{self.ollama_host}/api/tags", timeout=2)
            resp.raise_for_status()
            log.debug("Connected to Ollama at %s", self.ollama_host)
        except Exception as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.ollama_host}. Is the daemon running?"
            ) from exc

    def _setup_openai(self) -> None:
        """Initialize OpenAI provider with API key validation."""
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass openai_api_key parameter."
            )
        
        self._session.headers.update({
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        })
        log.debug("Configured OpenAI client for model %s", self.model)

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def __call__(self, image: Any) -> dict[str, str]:
        """
        Generate caption for image using the configured provider.
        
        Returns
        -------
        dict
            ``{'caption': str}`` containing the generated description.
        """
        if self.provider == "ollama":
            return self._call_ollama(image)
        elif self.provider == "openai":
            return self._call_openai(image)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    # --------------------------------------------------------------------- #
    # Provider implementations                                              #
    # --------------------------------------------------------------------- #
    def _call_ollama(self, image: Any) -> dict[str, str]:
        """Generate caption using Ollama provider."""
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

        log.debug("POST %s/api/generate [provider=ollama, model=%s]", self.ollama_host, self.model)
        resp = self._session.post(f"{self.ollama_host}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        caption = data.get("response", "").strip()

        if not caption:
            log.warning("Empty caption from Ollama!")
        return {"caption": caption}

    def _call_openai(self, image: Any) -> dict[str, str]:
        """Generate caption using OpenAI provider."""
        img_data_url = self._to_data_url(image)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "image_url", "image_url": {"url": img_data_url}}
                    ]
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        log.debug("POST %s/responses [provider=openai, model=%s]", self.openai_base_url, self.model)
        with open("/Users/akshayranganath/Downloads/payload.json", "w") as f:
            json.dump(payload, f)
        #resp = self._session.post(f"{self.openai_base_url}/chat/completions", json=payload, timeout=120)
        resp = self._session.post(f"{self.openai_base_url}/responses", json=payload, timeout=120)        
        resp.raise_for_status()
        data = resp.json()
        
        try:
            caption = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            log.error("Unexpected OpenAI response format: %s", data)
            raise RuntimeError("Failed to parse OpenAI response") from exc

        if not caption:
            log.warning("Empty caption from OpenAI!")
        return {"caption": caption}

    # --------------------------------------------------------------------- #
    # Image processing helpers                                              #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _to_base64_png(x: Any) -> str:
        """Convert image to base64-encoded PNG (for Ollama)."""
        img = Captioner._normalize_image(x)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _to_data_url(x: Any) -> str:
        """Convert image to data URL (for OpenAI)."""
        img = Captioner._normalize_image(x)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    @staticmethod
    def _normalize_image(x: Any) -> Image.Image:
        """Convert various image types to PIL.Image."""
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        elif isinstance(x, np.ndarray):
            if x.ndim == 2:  # grayscale → RGB
                x = np.stack([x] * 3, axis=-1)
            return Image.fromarray(x.astype("uint8"), mode="RGB")
        elif isinstance(x, torch.Tensor):
            if x.max() <= 1.0:
                x = x * 255
            x = x.clip(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            return Image.fromarray(x, mode="RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(x)}")


# --------------------------------------------------------------------------- #
# Manual test (python -m photo_indexer.models.captioner <image>)              #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python -m photo_indexer.models.captioner <image> [provider]")
        print("  provider: 'ollama' (default) or 'openai'")
        sys.exit(1)

    img_path = Path(sys.argv[1]).expanduser()
    provider = sys.argv[2] if len(sys.argv) > 2 else "ollama"
    
    cap = Captioner(provider=provider)
    result = cap(Image.open(img_path))
    print(f"[{provider}] {result}")
