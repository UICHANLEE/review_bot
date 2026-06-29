"""Map a receipt item name (text) to category candidates.

Phase 1 baseline needs no training data: Korean keyword matching against the
taxonomy, with an optional embedding fallback (CLIP/sentence text encoder) for
items that match no keyword. A trained classifier (Phase 3) can later replace
the `classify` body while keeping the same interface.
"""

from __future__ import annotations

from ..categories import Taxonomy
from ..types import CategoryScore


class TextCategoryClassifier:
    def __init__(self, taxonomy: Taxonomy, embedder=None, fallback_id: str = "etc"):
        self.taxonomy = taxonomy
        self.embedder = embedder
        self.fallback_id = fallback_id
        if embedder is not None:
            self._build_text_bank()

    def _build_text_bank(self) -> None:
        self._cat_ids = self.taxonomy.ids
        anchors = []
        for cid in self._cat_ids:
            c = self.taxonomy.by_id(cid)
            terms = [c.name_ko, *c.keywords, *c.aliases]
            anchors.append(", ".join(terms))
        self._text_bank = self.embedder.embed_texts(anchors)

    def _keyword_scores(self, item_name: str) -> dict[str, int]:
        low = item_name.lower()
        scores: dict[str, int] = {}
        for c in self.taxonomy:
            hits = sum(1 for kw in c.keywords if kw and kw.lower() in low)
            if hits:
                scores[c.id] = hits
        return scores

    def _embedding_scores(self, item_name: str, top_k: int) -> list[CategoryScore]:
        torch = self.embedder._torch
        emb = self.embedder.embed_texts([item_name])  # (1, D)
        sims = (emb @ self._text_bank.T)[0]
        probs = torch.softmax(sims / 0.05, dim=-1)
        k = min(top_k, len(self._cat_ids))
        vals, idx = probs.topk(k)
        return [
            CategoryScore(self._cat_ids[j], self.taxonomy.name_ko(self._cat_ids[j]), float(v))
            for v, j in zip(vals.tolist(), idx.tolist())
        ]

    def classify(self, item_name: str, top_k: int = 3) -> list[CategoryScore]:
        kw = self._keyword_scores(item_name)
        if kw:
            total = sum(kw.values())
            ranked = sorted(kw.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            return [
                CategoryScore(cid, self.taxonomy.name_ko(cid), hits / total)
                for cid, hits in ranked
            ]
        if self.embedder is not None:
            return self._embedding_scores(item_name, top_k)
        return [CategoryScore(self.fallback_id, self.taxonomy.name_ko(self.fallback_id), 0.0)]
