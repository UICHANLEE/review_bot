"""Build a single traceable image->logits module for on-device export.

Combines the (frozen) CLIP image encoder with the trained classifier head so the
exported Core ML / TFLite graph takes a preprocessed image tensor and returns
category logits directly. Swap the CLIP backbone for MobileCLIP before exporting
to get a phone-sized encoder; the wrapper is identical.
"""

from __future__ import annotations


def build_image_logits_module(embedder, head):
    """Return an nn.Module: (1,3,H,W) normalized tensor -> (1, num_classes) logits."""
    import torch.nn as nn

    class ImageLogits(nn.Module):
        def __init__(self, clip_model, head):
            super().__init__()
            self.clip_model = clip_model
            self.head = head

        def forward(self, pixel_values):
            feats = self.clip_model.get_image_features(pixel_values=pixel_values)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return self.head(feats)

    return ImageLogits(embedder.model, head).eval()


def example_input(image_size: int = 224):
    import torch

    return torch.randn(1, 3, image_size, image_size)
