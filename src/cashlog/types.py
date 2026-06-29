"""Shared data structures returned by the recognition pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CategoryScore:
    """A single category candidate with a normalized confidence score (0..1)."""

    category_id: str
    name_ko: str
    score: float


@dataclass
class ProductItem:
    """One recognized product and its ranked category candidates."""

    name: str
    raw_text: str | None = None
    categories: list[CategoryScore] = field(default_factory=list)

    @property
    def top_category(self) -> CategoryScore | None:
        return self.categories[0] if self.categories else None


@dataclass
class RecognitionResult:
    """End-to-end result for a single input image."""

    input_type: str  # "receipt" | "product"
    items: list[ProductItem]
    source: str  # "ondevice" | "server_vlm"
    confidence: float
