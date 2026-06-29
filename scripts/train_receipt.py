"""Phase 3: train a lightweight receipt item-name -> category text classifier.

Uses char n-gram TF-IDF + logistic regression (scikit-learn). This is tiny,
fast, robust to Korean spacing/typos from OCR, and trivial to run on a server;
for fully on-device text classification it can be reimplemented as a small
embedding+linear model and exported alongside the vision head.

Flattens every (item.name -> category_id) pair from receipt records.

Usage:
    python scripts/train_receipt.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cashlog.config import ARTIFACTS_DIR, LABELED_DIR  # noqa: E402
from cashlog.dataset import load_labels  # noqa: E402


def _pairs(records, split):
    xs, ys = [], []
    for r in records:
        if r.input_type != "receipt" or (r.split or "train") != split:
            continue
        for it in r.items:
            if it.name and it.category_id:
                xs.append(it.name)
                ys.append(it.category_id)
    return xs, ys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(LABELED_DIR / "labels.jsonl"))
    ap.add_argument("--out", default=str(ARTIFACTS_DIR / "receipt_text_clf.joblib"))
    args = ap.parse_args()

    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except ImportError:
        print("Install training extras: pip install -e .[train]")
        return

    records = load_labels(args.labels)
    Xtr, ytr = _pairs(records, "train")
    if not Xtr:
        print("No receipt training pairs. Run bootstrap_label.py + split_dataset.py first.")
        return

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, C=4.0)),
        ]
    )
    model.fit(Xtr, ytr)

    for split in ("val", "test"):
        Xs, ys = _pairs(records, split)
        if Xs:
            acc = model.score(Xs, ys)
            print(f"{split}: top1={acc:.3f} (n={len(ys)})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print(f"Saved receipt text classifier to {args.out}")


if __name__ == "__main__":
    main()
