"""Product-photo track: image embedding + zero-shot / trained category matching."""

from .embedder import ClipEmbedder
from .zeroshot import ZeroShotProductClassifier

__all__ = ["ClipEmbedder", "ZeroShotProductClassifier"]
