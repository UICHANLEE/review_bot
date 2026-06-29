"""Demo entrypoint: recognize products + recommend categories for one image.

Usage:
    python scripts/demo_image.py path/to/image.jpg [--type receipt|product] [--no-vlm]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--type", choices=["receipt", "product"], default=None)
    ap.add_argument("--no-vlm", action="store_true", help="disable server VLM fallback")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    from PIL import Image

    from cashlog.categories import Taxonomy
    from cashlog.pipeline import CashlogPipeline
    from cashlog.product.zeroshot import ZeroShotProductClassifier
    from cashlog.receipt.ocr import get_ocr_backend
    from cashlog.receipt.text_classifier import TextCategoryClassifier
    from cashlog.router import Router

    taxonomy = Taxonomy.load()
    product_clf = ZeroShotProductClassifier(taxonomy)
    text_clf = TextCategoryClassifier(taxonomy)
    ocr = get_ocr_backend("auto")

    pipeline = CashlogPipeline(
        taxonomy=taxonomy,
        product_classifier=product_clf,
        text_classifier=text_clf,
        ocr_backend=ocr,
        router=Router(ocr_backend=ocr),
    )

    image = Image.open(args.image).convert("RGB")
    result = pipeline.recognize(
        image, hint=args.type, top_k=args.top_k, allow_vlm_fallback=not args.no_vlm
    )

    print(f"input_type={result.input_type} source={result.source} confidence={result.confidence:.2f}")
    for item in result.items:
        cats = ", ".join(f"{c.name_ko}({c.score:.2f})" for c in item.categories)
        print(f"  - {item.name}: {cats}")


if __name__ == "__main__":
    main()
