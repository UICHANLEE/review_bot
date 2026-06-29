"""Phase 2: assign train/val/test splits to labeled records.

Prefers reviewed records for the val/test (evaluation) splits so metrics are
measured against human-verified labels. Writes the split back into the JSONL.

Usage:
    python scripts/split_dataset.py [--val 0.1 --test 0.1 --seed 42]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cashlog.config import LABELED_DIR  # noqa: E402
from cashlog.dataset import load_labels, save_labels  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(LABELED_DIR / "labels.jsonl"))
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = load_labels(args.path)
    if not records:
        print(f"No records in {args.path}. Run bootstrap_label.py first.")
        return

    rng = random.Random(args.seed)
    reviewed = [r for r in records if r.reviewed]
    unreviewed = [r for r in records if not r.reviewed]
    rng.shuffle(reviewed)
    rng.shuffle(unreviewed)

    # Put human-reviewed records first so the val/test (eval) splits are trustworthy.
    pool = reviewed + unreviewed

    n = len(pool)
    n_val = int(n * args.val)
    n_test = int(n * args.test)
    for i, r in enumerate(pool):
        if i < n_test:
            r.split = "test"
        elif i < n_test + n_val:
            r.split = "val"
        else:
            r.split = "train"

    save_labels(records, args.path)
    counts = {s: sum(1 for r in records if r.split == s) for s in ("train", "val", "test")}
    print(f"Split {n} records -> {counts}")
    print(f"  reviewed: {len(reviewed)}, unreviewed: {len(unreviewed)}")


if __name__ == "__main__":
    main()
