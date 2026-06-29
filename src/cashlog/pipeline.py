"""End-to-end orchestration: image -> recognized products -> category recommendation.

Routes the image to the receipt or product track, runs the on-device models, and
falls back to the server-side VLM when on-device confidence is below a threshold.
The VLM is loaded lazily so the on-device path has zero VLM overhead.
"""

from __future__ import annotations

from .categories import Taxonomy
from .router import Router
from .types import CategoryScore, ProductItem, RecognitionResult


class CashlogPipeline:
    def __init__(
        self,
        taxonomy: Taxonomy | None = None,
        product_classifier=None,
        text_classifier=None,
        ocr_backend=None,
        router: Router | None = None,
        vlm=None,
        confidence_threshold: float = 0.35,
    ):
        self.taxonomy = taxonomy or Taxonomy.load()
        self.product_classifier = product_classifier
        self.text_classifier = text_classifier
        self.ocr_backend = ocr_backend
        self.router = router or Router(ocr_backend=ocr_backend)
        self._vlm = vlm
        self.confidence_threshold = confidence_threshold

    # --- tracks -----------------------------------------------------------
    def _run_product(self, image, top_k: int) -> list[ProductItem]:
        cats = self.product_classifier.classify(image, top_k=top_k)
        return [ProductItem(name="<product photo>", categories=cats)]

    def _run_receipt(self, image, top_k: int) -> list[ProductItem]:
        from .receipt.parser import parse_items

        lines = self.ocr_backend.read_lines(image)
        items = parse_items(lines)
        out: list[ProductItem] = []
        for name in items:
            cats = self.text_classifier.classify(name, top_k=top_k)
            out.append(ProductItem(name=name, raw_text=name, categories=cats))
        return out

    # --- VLM fallback -----------------------------------------------------
    @property
    def vlm(self):
        if self._vlm is None:
            from .vlm.qwen_vl import QwenVL

            self._vlm = QwenVL.load()
        return self._vlm

    def _run_vlm(self, image, input_type: str) -> RecognitionResult:
        labeled = self.vlm.label_image(image, self.taxonomy, input_type=input_type)
        items = []
        for it in labeled.get("items", []):
            cid = it.get("category_id", "etc")
            try:
                name_ko = self.taxonomy.name_ko(cid)
            except KeyError:
                cid, name_ko = "etc", self.taxonomy.name_ko("etc")
            items.append(
                ProductItem(
                    name=it.get("name", ""),
                    categories=[CategoryScore(cid, name_ko, 1.0)],
                )
            )
        return RecognitionResult(
            input_type=labeled.get("input_type", input_type),
            items=items,
            source="server_vlm",
            confidence=1.0,
        )

    @staticmethod
    def _confidence(items: list[ProductItem]) -> float:
        tops = [it.top_category.score for it in items if it.top_category]
        return min(tops) if tops else 0.0

    # --- public API -------------------------------------------------------
    def recognize(
        self,
        image,
        hint: str | None = None,
        top_k: int = 3,
        allow_vlm_fallback: bool = True,
    ) -> RecognitionResult:
        input_type = self.router.route(image, hint=hint)

        if input_type == "receipt":
            items = self._run_receipt(image, top_k)
        else:
            items = self._run_product(image, top_k)

        confidence = self._confidence(items)
        if allow_vlm_fallback and (not items or confidence < self.confidence_threshold):
            return self._run_vlm(image, input_type)

        return RecognitionResult(
            input_type=input_type,
            items=items,
            source="ondevice",
            confidence=confidence,
        )
