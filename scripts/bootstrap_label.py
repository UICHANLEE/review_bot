"""Phase 2: auto-label raw images with Qwen2.5-VL to bootstrap a dataset.

Walks data/raw/, asks the VLM to recognize products + assign categories, and
appends records (source="vlm", reviewed=False) to data/labeled/labels.jsonl.
Already-labeled images are skipped so the script is resumable. A human then
reviews/corrects the records and flips "reviewed" to true (see review notes).

Usage:
    python scripts/bootstrap_label.py [--limit N] [--out data/labeled/labels.jsonl]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from cashlog.categories import Taxonomy  # noqa: E402
from cashlog.config import LABELED_DIR, RAW_DIR  # noqa: E402
from cashlog.dataset import (  # noqa: E402
    LabelItem,
    LabelRecord,
    iter_images,
    load_labels,
    rel_to_data,
    save_labels,
)
from cashlog.vlm import QwenVL  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW_DIR))
    ap.add_argument("--out", default=str(LABELED_DIR / "labels.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    taxonomy = Taxonomy.load()
    valid_ids = set(taxonomy.ids)

    records = load_labels(args.out)
    done = {r.image for r in records}

    images = iter_images(args.raw)
    todo = [p for p in images if rel_to_data(p) not in done]
    if args.limit:
        todo = todo[: args.limit]

    if not todo:
        print(f"Nothing to label ({len(images)} images, {len(done)} already done).")
        return

    print(f"Labeling {len(todo)} image(s) with Qwen2.5-VL...")
    vlm = QwenVL.load()

    for i, img_path in enumerate(todo, 1):
        try:
            image = Image.open(img_path).convert("RGB")
            labeled = vlm.label_image(image, taxonomy)
            items = [
                LabelItem(
                    name=it.get("name", ""),
                    category_id=it["category_id"] if it.get("category_id") in valid_ids else "etc",
                )
                for it in labeled.get("items", [])
            ]
            rec = LabelRecord(
                image=rel_to_data(img_path),
                input_type=labeled.get("input_type", "unknown"),
                items=items,
                source="vlm",
                reviewed=False,
            )
            records.append(rec)
            print(f"  [{i}/{len(todo)}] {rec.image}: {len(items)} item(s)")
        except Exception as exc:  # keep going; log the failure
            print(f"  [{i}/{len(todo)}] {img_path.name} FAILED: {exc}")

        save_labels(records, args.out)  # checkpoint after each image

    print(f"Done. Wrote {len(records)} records to {args.out}")
    print("Next: review records, correct categories, set \"reviewed\": true, then run split_dataset.py")


if __name__ == "__main__":
    main()
