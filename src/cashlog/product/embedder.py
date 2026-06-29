"""CLIP image/text embedder used for zero-shot category matching and distillation.

Uses the transformers CLIP implementation so it runs today with the installed
stack (torch + transformers) on Apple Silicon (MPS). For the actual on-device
build, swap the backbone for a MobileCLIP checkpoint and export via Core ML /
TFLite (see scripts/export_*.py) - the rest of the pipeline is unchanged.
"""

from __future__ import annotations

from typing import Sequence

from ..config import DEFAULT_CLIP_MODEL, pick_device


class ClipEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL,
        device: str | None = None,
    ):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = pick_device(device)
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name).eval().to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self._torch = torch

    @property
    def embed_dim(self) -> int:
        return self.model.config.projection_dim

    def embed_images(self, images: Sequence) -> "object":
        """Return L2-normalized image embeddings, shape (N, embed_dim)."""
        torch = self._torch
        inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        return torch.nn.functional.normalize(feats, dim=-1)

    def embed_texts(self, texts: Sequence[str]) -> "object":
        """Return L2-normalized text embeddings, shape (N, embed_dim)."""
        torch = self._torch
        inputs = self.processor(
            text=list(texts), return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        return torch.nn.functional.normalize(feats, dim=-1)
