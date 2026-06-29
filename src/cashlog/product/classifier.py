"""Compact on-device product classifier: a small head over frozen CLIP features.

Distillation strategy (Phase 3): keep the CLIP/MobileCLIP image encoder frozen
and train a tiny linear (or 1-hidden-layer) head on the bootstrapped dataset.
This head is cheap to train, tiny to ship, and exports cleanly to Core ML /
TFLite together with a MobileCLIP encoder.

The module also exposes `classify` so a trained head is a drop-in replacement
for `ZeroShotProductClassifier` inside the pipeline.
"""

from __future__ import annotations

from pathlib import Path

from ..categories import Taxonomy
from ..types import CategoryScore
from .embedder import ClipEmbedder


def build_head(in_dim: int, num_classes: int, hidden: int = 0):
    import torch.nn as nn

    if hidden > 0:
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )
    return nn.Linear(in_dim, num_classes)


class ProductHeadClassifier:
    def __init__(self, taxonomy: Taxonomy, embedder: ClipEmbedder, head):
        self.taxonomy = taxonomy
        self.embedder = embedder
        self.head = head.to(embedder.device).eval()
        self._ids = taxonomy.ids

    def classify(self, image, top_k: int = 3) -> list[CategoryScore]:
        return self.classify_batch([image], top_k=top_k)[0]

    def classify_batch(self, images, top_k: int = 3) -> list[list[CategoryScore]]:
        torch = self.embedder._torch
        with torch.no_grad():
            feats = self.embedder.embed_images(images)
            logits = self.head(feats)
            probs = torch.softmax(logits, dim=-1)
        out = []
        k = min(top_k, len(self._ids))
        for row in probs:
            vals, idx = row.topk(k)
            out.append(
                [
                    CategoryScore(self._ids[j], self.taxonomy.name_ko(self._ids[j]), float(v))
                    for v, j in zip(vals.tolist(), idx.tolist())
                ]
            )
        return out

    def save(self, path: str | Path) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.head.state_dict(),
                "category_ids": self._ids,
                "clip_model": self.embedder.model_name,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, taxonomy: Taxonomy | None = None) -> "ProductHeadClassifier":
        import torch

        ckpt = torch.load(path, map_location="cpu")
        taxonomy = taxonomy or Taxonomy.load()
        embedder = ClipEmbedder(model_name=ckpt["clip_model"])
        head = build_head(embedder.embed_dim, len(ckpt["category_ids"]))
        head.load_state_dict(ckpt["state_dict"])
        return cls(taxonomy, embedder, head)
