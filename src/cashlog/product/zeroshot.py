"""Zero-shot product -> category classification via CLIP image/text similarity.

No training data required: each category is turned into a small bank of text
prompts, and an image is assigned to the category whose prompt embeddings are
most similar. This is the Phase 1 baseline for the product-photo track.
"""

from __future__ import annotations

from typing import Sequence

from ..categories import Taxonomy
from ..types import CategoryScore
from .embedder import ClipEmbedder


class ZeroShotProductClassifier:
    def __init__(
        self,
        taxonomy: Taxonomy,
        embedder: ClipEmbedder | None = None,
        prompt_template: str = "a photo of {label}",
        temperature: float = 0.01,
    ):
        self.taxonomy = taxonomy
        self.embedder = embedder or ClipEmbedder()
        self.prompt_template = prompt_template
        self.temperature = temperature
        self._build_text_bank()

    def _build_text_bank(self) -> None:
        """Precompute category text embeddings once (reused across all images)."""
        prompts = self.taxonomy.clip_prompts(self.prompt_template)
        self._cat_ids: list[str] = []
        self._prompt_slices: list[slice] = []
        flat: list[str] = []
        for cid in self.taxonomy.ids:
            ps = prompts[cid]
            start = len(flat)
            flat.extend(ps)
            self._prompt_slices.append(slice(start, len(flat)))
            self._cat_ids.append(cid)
        self._text_bank = self.embedder.embed_texts(flat)  # (P, D)

    def classify(self, image, top_k: int = 3) -> list[CategoryScore]:
        return self.classify_batch([image], top_k=top_k)[0]

    def classify_batch(self, images: Sequence, top_k: int = 3) -> list[list[CategoryScore]]:
        torch = self.embedder._torch
        img_emb = self.embedder.embed_images(images)  # (N, D)
        sims = img_emb @ self._text_bank.T  # (N, P) cosine (already normalized)

        # Per-category score = max similarity over its prompts.
        n = sims.shape[0]
        cat_scores = torch.empty((n, len(self._cat_ids)), device=sims.device)
        for j, sl in enumerate(self._prompt_slices):
            cat_scores[:, j] = sims[:, sl].max(dim=1).values

        probs = torch.softmax(cat_scores / self.temperature, dim=-1)

        results: list[list[CategoryScore]] = []
        for i in range(n):
            row = probs[i]
            k = min(top_k, len(self._cat_ids))
            vals, idx = row.topk(k)
            results.append(
                [
                    CategoryScore(
                        category_id=self._cat_ids[j],
                        name_ko=self.taxonomy.name_ko(self._cat_ids[j]),
                        score=float(v),
                    )
                    for v, j in zip(vals.tolist(), idx.tolist())
                ]
            )
        return results
