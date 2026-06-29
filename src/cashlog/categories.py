"""Loading and querying the fixed expense-category taxonomy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import CATEGORIES_PATH


@dataclass(frozen=True)
class Category:
    id: str
    name_ko: str
    aliases: tuple[str, ...]
    keywords: tuple[str, ...]


class Taxonomy:
    """The set of categories the app supports, plus helpers for prompts/matching."""

    def __init__(self, categories: list[Category]):
        if not categories:
            raise ValueError("Taxonomy must contain at least one category")
        self.categories = categories
        self._by_id = {c.id: c for c in categories}

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Taxonomy":
        path = Path(path) if path else CATEGORIES_PATH
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        categories = [
            Category(
                id=c["id"],
                name_ko=c["name_ko"],
                aliases=tuple(c.get("aliases", [])),
                keywords=tuple(c.get("keywords", [])),
            )
            for c in data["categories"]
        ]
        return cls(categories)

    def __len__(self) -> int:
        return len(self.categories)

    def __iter__(self):
        return iter(self.categories)

    @property
    def ids(self) -> list[str]:
        return [c.id for c in self.categories]

    def by_id(self, category_id: str) -> Category:
        return self._by_id[category_id]

    def name_ko(self, category_id: str) -> str:
        return self._by_id[category_id].name_ko

    def clip_prompts(self, template: str = "a photo of {label}") -> dict[str, list[str]]:
        """Return {category_id: [prompt, ...]} for CLIP zero-shot scoring.

        Each alias becomes its own prompt; the per-category score is the max over
        its prompts. The Korean name is also included so multilingual CLIP
        checkpoints can use it directly.
        """
        prompts: dict[str, list[str]] = {}
        for c in self.categories:
            labels = list(c.aliases) or [c.name_ko]
            labels.append(c.name_ko)
            prompts[c.id] = [template.format(label=lbl) for lbl in labels]
        return prompts
