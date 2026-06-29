"""Phase 4: export the product image->logits model to TFLite (Android).

Uses ai-edge-torch (Google's PyTorch->TFLite path) when available. Produces a
quantized .tflite that runs via NNAPI / GPU delegate on Android. Use a
MobileCLIP backbone for a phone-sized model.

Usage:
    python scripts/export_tflite.py [--head artifacts/product_head.pt]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cashlog.config import ARTIFACTS_DIR  # noqa: E402
from cashlog.product.classifier import ProductHeadClassifier  # noqa: E402
from cashlog.product.export import build_image_logits_module, example_input  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=str(ARTIFACTS_DIR / "product_head.pt"))
    ap.add_argument("--out", default=str(ARTIFACTS_DIR / "product_classifier.tflite"))
    ap.add_argument("--image-size", type=int, default=224)
    args = ap.parse_args()

    if not Path(args.head).exists():
        print(f"Missing {args.head}. Train it first: python scripts/train_product.py")
        return

    try:
        import ai_edge_torch  # type: ignore
    except ImportError:
        print(
            "ai-edge-torch not installed. Install with:\n"
            "  pip install ai-edge-torch\n"
            "(Linux/x86 recommended; on macOS use the ONNX->TF->TFLite path instead.)"
        )
        return

    clf = ProductHeadClassifier.load(args.head)
    module = build_image_logits_module(clf.embedder, clf.head).cpu()
    ex = (example_input(args.image_size),)

    edge_model = ai_edge_torch.convert(module, ex)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(args.out)
    print(f"Saved TFLite model to {args.out}")
    print("Category order (index -> id):")
    for i, cid in enumerate(clf._ids):
        print(f"  {i}: {cid}")


if __name__ == "__main__":
    main()
