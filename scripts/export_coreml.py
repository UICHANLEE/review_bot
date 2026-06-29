"""Phase 4: export the product image->logits model to Core ML (iOS).

Loads the trained head (artifacts/product_head.pt), wraps it with the CLIP image
encoder, traces it, and converts to a quantized .mlpackage. Use a MobileCLIP
backbone for a phone-sized model.

Usage:
    python scripts/export_coreml.py [--head artifacts/product_head.pt] [--quantize int8]
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
    ap.add_argument("--out", default=str(ARTIFACTS_DIR / "ProductClassifier.mlpackage"))
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--quantize", choices=["none", "int8", "fp16"], default="int8")
    args = ap.parse_args()

    if not Path(args.head).exists():
        print(f"Missing {args.head}. Train it first: python scripts/train_product.py")
        return

    try:
        import coremltools as ct
        import torch
    except ImportError:
        print("Install export extras: pip install -e .[export]")
        return

    clf = ProductHeadClassifier.load(args.head)
    module = build_image_logits_module(clf.embedder, clf.head).cpu()
    ex = example_input(args.image_size)
    traced = torch.jit.trace(module, ex)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="pixel_values", shape=ex.shape)],
        classifier_config=ct.ClassifierConfig(class_labels=list(clf._ids)),
        minimum_deployment_target=ct.target.iOS16,
    )

    if args.quantize == "int8":
        from coremltools.optimize.coreml import (
            OpLinearQuantizerConfig,
            OptimizationConfig,
            linear_quantize_weights,
        )

        cfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear", dtype="int8"))
        mlmodel = linear_quantize_weights(mlmodel, config=cfg)
    elif args.quantize == "fp16":
        mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(mlmodel, nbits=16)

    mlmodel.save(args.out)
    print(f"Saved Core ML model to {args.out} (quantize={args.quantize})")


if __name__ == "__main__":
    main()
