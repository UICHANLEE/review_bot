"""Read/write helpers for the bootstrapped label store (JSONL).

Each record describes one image:
    {
      "image": "raw/img_001.jpg",      # path relative to data/
      "input_type": "receipt|product",
      "items": [{"name": str, "category_id": str}],
      "source": "vlm|human",
      "reviewed": bool,
      "split": "train|val|test|null"
    }
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import DATA_DIR

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class LabelItem:
    name: str
    category_id: str


@dataclass
class LabelRecord:
    image: str
    input_type: str = "unknown"
    items: list[LabelItem] = field(default_factory=list)
    source: str = "vlm"
    reviewed: bool = False
    split: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LabelRecord":
        return cls(
            image=d["image"],
            input_type=d.get("input_type", "unknown"),
            items=[LabelItem(**it) for it in d.get("items", [])],
            source=d.get("source", "vlm"),
            reviewed=d.get("reviewed", False),
            split=d.get("split"),
        )


def iter_images(root: Path | str) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def load_labels(path: Path | str) -> list[LabelRecord]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [LabelRecord.from_dict(json.loads(ln)) for ln in fh if ln.strip()]


def save_labels(records: list[LabelRecord], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def rel_to_data(p: Path) -> str:
    p = Path(p)
    try:
        return str(p.relative_to(DATA_DIR))
    except ValueError:
        return str(p)
