# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NTU IE4483 mini project (Option 1), due **13 Nov 2026**: binary sentiment classification of product reviews (`1` = positive, `0` = negative; Blitzer et al. 2007 data), done by a group of 3. Three models are compared: TF-IDF + classifier, word embeddings (GloVe) + RNN/BiLSTM, and fine-tuned BERT. The best model produces the final `submission.csv` (1,851 rows, same columns and row order as the template). The deliverable is also a written report answering questions (a)–(h).

`context/` holds the source of truth: `IE4483-Project1.pdf` (assignment brief) and `IE4483_Sentiment_Project_Plan.pdf` (team plan: model details, tuning protocol, timeline, report structure).

## Repository layout and how it runs

There is no build system, test suite, linter, or `requirements.txt`. Notebook 00 needs `pandas` and `scikit-learn`; run it from the repo root with `jupyter nbconvert --to notebook --execute --inplace 00_Data_Preparation.ipynb`. It is deterministic: re-running produces byte-identical output files. All code is **Jupyter notebooks**. `00_Data_Preparation.ipynb` is local; the model notebooks are written for Google Colab (`03_BERT.ipynb` calls `drive.mount`, uses `/content/drive/...` paths and asserts a GPU), so it is not meant to run locally as-is.

- `00_Data_Preparation.ipynb`: run once. Runs locally (reads `data/raw/`, no Colab/Drive). Cleans the text (HTML, entities, unicode, whitespace), drops empty/conflicting/duplicate train reviews before splitting, makes the stratified split, and writes `train_split.json`, `val_split.json`, `test_split.json` (reviews only, original order) to `data/processed/`.
- `03_BERT.ipynb`: BERT model notebook (Hugging Face `Trainer`, `bert-base-uncased`, `max_length=128`). It defines `train_and_evaluate(...)`, runs a dict of one-variable-at-a-time experiments against a `baseline` config, writes per-run `metrics.json` plus a `comparison.csv`, and copies the best run's `submission.csv`. Outputs go to Drive under `IE4483_BERT/experiments/<run>/`.
- `data/raw/`: local `train.json` and the `submission.csv` template (all predictions 0). There is no `test.json`; the test set is the unlabeled reviews in `submission.csv`.
- `data/processed/`: output of notebook 00 (`train_split.json`, `val_split.json`, `test_split.json`, `prep_summary.json`).
- The TF-IDF and RNN notebooks (presumably `01_*`, `02_*` by the numbering) are not in the repo yet; teammates own them.

## Conventions every model notebook must follow

- **Never re-split or re-clean the data.** Load `train_split.json` / `val_split.json` / `test_split.json` from `data/processed/` (`03_BERT.ipynb` still reads them from a Drive `SHARED_DIR`, so upload the regenerated files there or update its paths). The split is stratified **85/15, seed 42**.
- Data columns: `reviews` (cleaned text) and `sentiments` (0/1; absent in the test split). The submission column is `predictions`.
- Tune on the validation set only; the test set is never used for tuning.
- Report the same metrics for all models so they can be compared: accuracy, precision/recall, F1 and **Negative F1** (classes are imbalanced), confusion matrix, PR-AUC (needs probabilities), error analysis, training time, and training curves (for the neural models).
- Final numbers should come from multiple seeds (plan: 3 seeds, mean ± std); a single global seed is set at the top of each notebook.
- Fit anything data-dependent (TF-IDF vocab, GloVe vocab) on the training split only.

## Known inconsistencies to be aware of

- The plan PDF says an 80/20 split and macro-F1; the README and `00_Data_Preparation.ipynb` use 85/15 and Negative F1. The README and notebook are authoritative.
- `03_BERT.ipynb` runs its confusion matrix, PR curve, and error-analysis helpers on `test_enc` / the test data and reads `labels` from them, but there is no labeled test set (the test set is the unlabeled reviews in `submission.csv`), so those cells need to use the validation set instead.
