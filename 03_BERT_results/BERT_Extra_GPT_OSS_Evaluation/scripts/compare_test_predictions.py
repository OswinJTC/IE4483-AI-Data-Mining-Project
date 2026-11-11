#!/usr/bin/env python3
"""Compare saved Run 10 predictions with GPT-OSS 120B pseudo-labels."""
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BERT = ROOT.parent / "BERT_Stage6_Freezing_and_Early_Stopping/Run_10_EarlyStopping/submission.csv"
REFERENCE = ROOT / "test" / "submission_actual_result.csv"
REPORT = ROOT / "test" / "test_evaluation.txt"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    reference, predictions = read_csv(REFERENCE), read_csv(BERT)
    if len(reference) != 1851 or len(predictions) != 1851:
        raise ValueError("Both files must contain exactly 1,851 reviews.")
    counts = Counter()
    for i, (ref, pred) in enumerate(zip(reference, predictions), 1):
        if ref["reviews"] != pred["reviews"]:
            raise ValueError(f"Review text or order differs at row {i}.")
        pair = (ref["predictions"], pred["predictions"])
        if any(label not in {"0", "1"} for label in pair):
            raise ValueError(f"Expected binary labels at row {i}.")
        counts[pair] += 1

    nn, np, pn, pp = (counts[p] for p in (("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")))
    correct = nn + pp
    f1 = 2 * nn / (2 * nn + np + pn) if nn + np + pn else 0.0
    report = (
        "BERT Run 10 vs. GPT-OSS 120B pseudo-labels\n"
        "1,851 test reviews; text and row order verified.\n"
        "Reference: GPT-OSS 120B pseudo-labels (not human ground truth).\n\n"
        f"Accuracy against pseudo-labels: {correct / len(reference):.2%} ({correct:,}/{len(reference):,})\n"
        f"Negative F1: {f1:.4f}\n"
        f"Disagreements: {np + pn}\n\n"
        "Confusion matrix (rows: GPT-OSS; columns: BERT)\n"
        "                   BERT 0  BERT 1\n"
        f"GPT-OSS 0 (Negative) {nn:5d}   {np:5d}\n"
        f"GPT-OSS 1 (Positive) {pn:5d}   {pp:5d}\n"
    )
    if REPORT.exists() and REPORT.read_bytes() != report.encode("utf-8"):
        raise ValueError("Recomputed metrics differ from the archived report; the report was not changed.")
    if not REPORT.exists():
        REPORT.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
