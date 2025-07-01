"""
photo_indexer.models.captioner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Flexible vision-language captioning client that supports local (Ollama, BLIP-2) and 
remote (OpenAI) providers for generating image descriptions.

Key points
----------
* **Ollama provider**: Supports local models like llama3.2-vision:latest
* **OpenAI provider**: Supports GPT-4 Vision models (gpt-4-vision-preview, gpt-4o, etc.)
* **BLIP-2 provider**: Lightweight Hugging Face model (Salesforce/blip2-opt-2.7b)
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

ProviderType = Literal["ollama", "openai", "blip2"]


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
        "Do not start with generic descriptions like 'a photo of' or 'a picture of'."
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
            Choose between "ollama" (local), "openai" (remote), or "blip2" (local).
        model:
            Model name. Defaults:
            - Ollama: "llama3.2-vision:latest"  
            - OpenAI: "gpt-4o"
            - BLIP-2: "Salesforce/blip2-opt-2.7b"
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
        elif provider == "blip2":
            self.model = model or "Salesforce/blip2-opt-2.7b"
            self._setup_blip2()
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
        
        # Mask API key for logging (show first 10 and last 4 characters)
        masked_key = f"{self.openai_api_key[:10]}...{self.openai_api_key[-4:]}"
        log.debug("Setting up OpenAI client with API key: %s", masked_key)
        
        self._session.headers.update({
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        })
        log.debug("Configured OpenAI client for model %s", self.model)

    def _setup_blip2(self) -> None:
        """Initialize BLIP-2 provider with model loading."""
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            log.debug("Loading BLIP-2 model: %s", self.model)
            
            # Load model and processor - use CPU for better compatibility
            self.blip2_processor = Blip2Processor.from_pretrained(self.model)
            self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
                self.model, 
                torch_dtype=torch.float32,  # Use float32 for CPU compatibility
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Move to appropriate device
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            
            self.blip2_device = device
            if device != "cuda":  # device_map="auto" handles CUDA placement
                self.blip2_model = self.blip2_model.to(device)
            
            log.info("Loaded BLIP-2 model %s on device: %s", self.model, device)
            
        except ImportError as exc:
            raise RuntimeError(
                "BLIP-2 provider requires transformers library. "
                "Install with: pip install transformers"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load BLIP-2 model {self.model}") from exc

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
        elif self.provider == "blip2":
            return self._call_blip2(image)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    # --------------------------------------------------------------------- #
    # Provider implementations                                              #
    # --------------------------------------------------------------------- #
    def _call_ollama(self, image: Any) -> dict[str, str]:
        """Generate caption using Ollama provider."""
        img_b64 = self._to_base64_png(image)
        '''
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
        '''
        # removing the options field from payload. It was causing a strange error that appeared to make the llama not work.
        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "images": [img_b64],
            "stream": False         
        }
        log.debug("POST %s/api/generate [provider=ollama, model=%s]", self.ollama_host, self.model)
        
        #with open("/Users/akshayranganath/Downloads/payload.json", "w") as f:
        #    json.dump(payload, f, indent=2)
        
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

        # Debug logging to verify API key is being sent
        headers_debug = dict(self._session.headers)
        if 'Authorization' in headers_debug:
            # Mask the API key for security in logs
            auth_header = headers_debug['Authorization']
            if auth_header.startswith('Bearer '):
                headers_debug['Authorization'] = f"Bearer {auth_header[7:17]}...{auth_header[-4:]}"
        
        log.debug("POST %s/chat/completions [provider=openai, model=%s]", self.openai_base_url, self.model)
        log.debug("Request headers: %s", headers_debug)
        
        # Optional: Save payload for debugging (remove in production)
        # with open("/Users/akshayranganath/Downloads/payload.json", "w") as f:
        #     json.dump(payload, f, indent=2)
        
        resp = self._session.post(f"{self.openai_base_url}/chat/completions", json=payload, timeout=120)        
        
        # Enhanced error handling for API key issues
        if resp.status_code == 401:
            log.error("OpenAI API authentication failed. Check your API key.")
            log.error("Response: %s", resp.text)
            raise RuntimeError("OpenAI API authentication failed - invalid or missing API key")
        elif resp.status_code == 403:
            log.error("OpenAI API access forbidden. Check your API key permissions.")
            raise RuntimeError("OpenAI API access forbidden")
        elif not resp.ok:
            log.error("OpenAI API request failed with status %d: %s", resp.status_code, resp.text)
            
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

    def _call_blip2(self, image: Any) -> dict[str, str]:
        """Generate caption using BLIP-2 provider."""
        img_pil = self._normalize_image(image)
        
        log.debug("Generating caption with BLIP-2 model %s", self.model)
        
        try:
            # Process image and generate caption
            inputs = self.blip2_processor(img_pil, return_tensors="pt").to(self.blip2_device)
            
            # Generate with specified parameters
            with torch.no_grad():
                generated_ids = self.blip2_model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=True if self.temperature > 0 else False,
                    temperature=max(self.temperature, 0.1),  # Avoid zero temperature
                    num_beams=3,  # Use beam search for better quality
                    early_stopping=True,
                )
            
            # Decode the generated caption
            caption = self.blip2_processor.decode(
                generated_ids[0], 
                skip_special_tokens=True
            ).strip()
            
            # Remove common prefixes that BLIP-2 sometimes adds
            prefixes_to_remove = [
                "a photo of ", "an image of ", "a picture of ",
                "the image shows ", "this image shows ", "there is "
            ]
            
            caption_lower = caption.lower()
            for prefix in prefixes_to_remove:
                if caption_lower.startswith(prefix):
                    caption = caption[len(prefix):]
                    break
            
            # Capitalize first letter
            if caption:
                caption = caption[0].upper() + caption[1:]
            
            if not caption:
                log.warning("Empty caption from BLIP-2!")
                caption = "Image content could not be described"
                
            return {"caption": caption}
            
        except Exception as exc:
            log.error("BLIP-2 inference failed: %s", exc)
            return {"caption": "Error generating caption"}

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
# API Key Test Function                                                        #
# --------------------------------------------------------------------------- #
def test_openai_api_key(api_key: str = None) -> bool:
    """
    Test if OpenAI API key is valid by making a simple request.
    
    Parameters
    ----------
    api_key : str, optional
        API key to test. If None, uses OPENAI_API_KEY env var.
        
    Returns
    -------
    bool
        True if API key is valid, False otherwise.
    """
    import requests
    
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No API key provided or found in OPENAI_API_KEY env var")
        return False
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Simple test request to list models
    try:
        resp = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=10)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            vision_models = [m["id"] for m in models if "gpt-4" in m["id"] and "vision" in m["id"] or "gpt-4o" in m["id"]]
            print(f"✅ API key is valid! Found {len(vision_models)} vision models: {vision_models[:3]}")
            return True
        elif resp.status_code == 401:
            print("❌ API key is invalid or expired")
            return False
        else:
            print(f"❌ API request failed with status {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing API key: {e}")
        return False


# --------------------------------------------------------------------------- #
# Manual test (python -m photo_indexer.models.captioner <image>)              #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python -m photo_indexer.models.captioner <image> [provider]")
        print("       python -m photo_indexer.models.captioner --test-api-key")
        print("  provider: 'ollama' (default), 'openai', or 'blip2'")
        sys.exit(1)

    # Test API key functionality
    if sys.argv[1] == "--test-api-key":
        test_openai_api_key()
        sys.exit(0)

    img_path = Path(sys.argv[1]).expanduser()
    provider = sys.argv[2] if len(sys.argv) > 2 else "ollama"
    
    cap = Captioner(provider=provider)
    result = cap(Image.open(img_path))
    print(f"[{provider}] {result}")
