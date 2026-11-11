#!/usr/bin/env python3
"""Label the unlabeled test split, one review per Groq request."""

import csv

from label_validation import EVALUATION, ROOT, label_reviews

DATA = ROOT / "data/processed/test_split.json"
CHECKPOINT = EVALUATION / "test" / ".test_progress.json"
TARGET = EVALUATION / "test" / "pseudo_labels.csv"


def main():
    if TARGET.exists():
        print(f"Saved test labels already exist: {TARGET}")
        return
    rows, labels = label_reviews(DATA, CHECKPOINT, 1851)
    target = TARGET
    temporary = target.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "reviews", "pseudo_label"])
        writer.writerows(
            (row_id, row["reviews"], label)
            for row_id, (row, label) in enumerate(zip(rows, labels), 1)
        )
    temporary.replace(target)
    print(f"Saved {len(labels):,} pseudo-labels: {labels.count(0):,} Negative, "
          f"{labels.count(1):,} Positive.\n{target}")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"Paused: {exc or 'interrupted'}. Saved rows will be skipped on restart.")
        raise SystemExit(2) from None
