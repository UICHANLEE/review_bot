"""Phase 3: train the compact product classifier head over frozen CLIP features.

Reads product-type records from data/labeled/labels.jsonl (each must have a
single item with a category_id), precomputes CLIP image embeddings, trains a
small head, evaluates top-1/top-3 on the val/test splits, and saves the head to
artifacts/product_head.pt.

Usage:
    python scripts/train_product.py [--epochs 50 --hidden 0 --lr 1e-3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from cashlog.categories import Taxonomy  # noqa: E402
from cashlog.config import ARTIFACTS_DIR, DATA_DIR, LABELED_DIR  # noqa: E402
from cashlog.dataset import load_labels  # noqa: E402
from cashlog.product.classifier import ProductHeadClassifier, build_head  # noqa: E402
from cashlog.product.embedder import ClipEmbedder  # noqa: E402


def _product_examples(records, taxonomy):
    id_to_idx = {cid: i for i, cid in enumerate(taxonomy.ids)}
    rows = []
    for r in records:
        if r.input_type != "product" or not r.items:
            continue
        cid = r.items[0].category_id
        if cid not in id_to_idx:
            continue
        rows.append((r.image, id_to_idx[cid], r.split or "train"))
    return rows


def _embed_split(rows, split, embedder):
    import torch

    paths = [p for p, _, s in rows if s == split]
    labels = [y for _, y, s in rows if s == split]
    if not paths:
        return None, None
    imgs = [Image.open(DATA_DIR / p).convert("RGB") for p in paths]
    feats = embedder.embed_images(imgs).cpu()
    return feats, torch.tensor(labels)


def _topk_acc(logits, y, k):
    import torch

    topk = logits.topk(min(k, logits.shape[1]), dim=1).indices
    return (topk == y.unsqueeze(1)).any(dim=1).float().mean().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(LABELED_DIR / "labels.jsonl"))
    ap.add_argument("--out", default=str(ARTIFACTS_DIR / "product_head.pt"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=0)
    args = ap.parse_args()

    import torch

    taxonomy = Taxonomy.load()
    records = load_labels(args.labels)
    rows = _product_examples(records, taxonomy)
    if not rows:
        print("No product examples found. Run bootstrap_label.py + split_dataset.py first.")
        return

    embedder = ClipEmbedder()
    Xtr, ytr = _embed_split(rows, "train", embedder)
    if Xtr is None:
        print("No training rows after split.")
        return

    head = build_head(embedder.embed_dim, len(taxonomy.ids), hidden=args.hidden)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    head.train()
    for ep in range(args.epochs):
        opt.zero_grad()
        loss = loss_fn(head(Xtr), ytr)
        loss.backward()
        opt.step()
        if (ep + 1) % 10 == 0:
            print(f"epoch {ep + 1}/{args.epochs} loss={loss.item():.4f}")

    head.eval()
    with torch.no_grad():
        for split in ("val", "test"):
            X, y = _embed_split(rows, split, embedder)
            if X is None:
                continue
            logits = head(X)
            print(
                f"{split}: top1={_topk_acc(logits, y, 1):.3f} "
                f"top3={_topk_acc(logits, y, 3):.3f} (n={len(y)})"
            )

    clf = ProductHeadClassifier(taxonomy, embedder, head)
    clf.save(args.out)
    print(f"Saved head to {args.out}")


if __name__ == "__main__":
    main()
