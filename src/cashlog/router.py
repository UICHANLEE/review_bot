"""Decide whether an input image is a receipt or a product photo.

The app can pass an explicit hint (e.g. the user tapped "scan receipt"). When no
hint is given we fall back to a cheap text-density heuristic over OCR lines: a
receipt has many short text lines with prices, a product photo has few.
"""

from __future__ import annotations

from .receipt.parser import parse_items


class Router:
    def __init__(self, ocr_backend=None, receipt_line_threshold: int = 3):
        self.ocr_backend = ocr_backend
        self.receipt_line_threshold = receipt_line_threshold

    def route(self, image, hint: str | None = None) -> str:
        if hint in ("receipt", "product"):
            return hint
        if self.ocr_backend is None:
            # Without OCR we cannot measure text density; default to product.
            return "product"
        lines = self.ocr_backend.read_lines(image)
        item_like = parse_items(lines)
        return "receipt" if len(item_like) >= self.receipt_line_threshold else "product"
