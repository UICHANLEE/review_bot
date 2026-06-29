"""Phase 4: benchmark on-device latency + accuracy on the dev machine (M1 Pro).

Measures per-image latency (p50/p90) for the product track and, if a labeled
test split exists, reports top-1/top-3 category accuracy. Use this to verify the
"fast enough on phone" target before/after export and to tune the confidence
threshold that triggers the server VLM fallback.

Usage:
    python scripts/benchmark.py [--mode zeroshot|head] [--runs 30] [--device mps]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from cashlog.categories import Taxonomy  # noqa: E402
from cashlog.config import ARTIFACTS_DIR, DATA_DIR, LABELED_DIR  # noqa: E402
from cashlog.dataset import load_labels  # noqa: E402


def _make_classifier(mode, device):
    from cashlog.product.embedder import ClipEmbedder

    taxonomy = Taxonomy.load()
    embedder = ClipEmbedder(device=device)
    if mode == "head":
        from cashlog.product.classifier import ProductHeadClassifier

        head_path = ARTIFACTS_DIR / "product_head.pt"
        if not head_path.exists():
            raise SystemExit(f"Missing {head_path}; train it or use --mode zeroshot")
        return ProductHeadClassifier.load(head_path, taxonomy)
    from cashlog.product.zeroshot import ZeroShotProductClassifier

    return ZeroShotProductClassifier(taxonomy, embedder)


def _sample_image():
    imgs = list(DATA_DIR.glob("raw/**/*"))
    imgs = [p for p in imgs if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if imgs:
        return Image.open(imgs[0]).convert("RGB")
    return Image.new("RGB", (224, 224), (127, 127, 127))


def _latency(clf, image, runs):
    clf.classify(image)  # warmup
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        clf.classify(image)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p = lambda q: times[min(len(times) - 1, int(len(times) * q))]
    return statistics.mean(times), p(0.5), p(0.9)


def _accuracy(clf):
    records = [r for r in load_labels(LABELED_DIR / "labels.jsonl")
               if r.input_type == "product" and (r.split == "test") and r.items]
    if not records:
        return None
    top1 = top3 = 0
    for r in records:
        gold = r.items[0].category_id
        cats = clf.classify(Image.open(DATA_DIR / r.image).convert("RGB"), top_k=3)
        ids = [c.category_id for c in cats]
        top1 += int(ids[:1] == [gold])
        top3 += int(gold in ids)
    n = len(records)
    return top1 / n, top3 / n, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["zeroshot", "head"], default="zeroshot")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--device", default=None, help="mps|cpu|cuda (default: auto)")
    args = ap.parse_args()

    clf = _make_classifier(args.mode, args.device)
    image = _sample_image()

    mean, p50, p90 = _latency(clf, image, args.runs)
    print(f"[{args.mode}] latency over {args.runs} runs: "
          f"mean={mean:.1f}ms p50={p50:.1f}ms p90={p90:.1f}ms")

    acc = _accuracy(clf)
    if acc:
        top1, top3, n = acc
        print(f"[{args.mode}] test accuracy: top1={top1:.3f} top3={top3:.3f} (n={n})")
    else:
        print("[accuracy] no labeled product test split; skipping.")


if __name__ == "__main__":
    main()
