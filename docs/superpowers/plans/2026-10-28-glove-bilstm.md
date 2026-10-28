# GloVe + BiLSTM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate the project's second model, a GloVe + BiLSTM (attention) sentiment classifier, on `data/processed/train_split.json` and `val_split.json`, with a one-variable-at-a-time experiment table.

**Architecture:** Reusable logic lives in `src/glove_bilstm.py` (data, vocab, GloVe, model, training, metrics) with pytest tests; `02_GloVe_BiLSTM.ipynb` is a thin driver that runs the experiments, plots curves and writes results. Everything runs locally; the test split is never touched.

**Tech Stack:** Python 3.11, PyTorch, numpy, scikit-learn, matplotlib, pytest, GloVe 6B 100d (`https://nlp.stanford.edu/data/glove.6B.zip`).

**Spec:** No separate design doc. The user skipped brainstorming questions; the decisions are the ones announced in chat and the Global Constraints below, which are the assumptions to review. Background: `context/IE4483_Sentiment_Project_Plan.pdf` (Model 2) and `README.md` (evaluation standard).

## Global Constraints

- Inputs: `data/processed/train_split.json` (train) and `val_split.json` (evaluation); columns `reviews` (str), `sentiments` (0 negative, 1 positive). Never re-split or re-clean; never read `test_split.json`.
- Seed 42 for the tuning runs; the best configuration is re-run with seeds 42, 43, 44 and reported as mean ± std.
- Vocabulary and the embedding matrix are built from the training split only.
- Tokenization: lowercase, regex `[a-z0-9']+|[^\sa-z0-9']` (words plus single punctuation marks, so "not" and "!" survive). Special ids: `<pad>`=0, `<unk>`=1.
- `max_len` = 150 (about the 95th percentile of review words; longer reviews are truncated, shorter ones right-padded).
- GloVe: 6B, 100 dimensions, downloaded to `data/glove/` which must be git-ignored.
- Model: `Embedding → BiLSTM (hidden 128, layers 1, dropout 0.4) → pooling (attention default) → Linear(2×hidden, 2)`; loss cross-entropy; optimizer AdamW, lr 1e-3, batch size 64, gradient clip 1.0, max 15 epochs, early stopping patience 3 on validation Negative F1.
- Metrics (per README): accuracy, precision, recall, F1, Negative F1, confusion matrix, PR-AUC (positive class), training time, error analysis (misclassified validation examples).
- Experiments change one variable from the baseline each: embeddings frozen (baseline) vs fine-tuned; pooling attention vs max vs last-state; class-weighted loss vs unweighted; unidirectional vs bidirectional; hidden size 64/128/256; lr 3e-4/1e-3/3e-3.
- Outputs go to `results/bilstm/` (`experiments.csv`, `best_final.json`, plots, `errors.csv`). Do not modify `00_Data_Preparation.ipynb`, `03_BERT.ipynb` or anything in `data/`.

## Review Focus

- A review that is empty after tokenization, or whose words are all out-of-vocabulary, encodes to a valid padded sequence (no crash, no NaN loss).
- A review longer than `max_len` is truncated to exactly `max_len` ids.
- Padding never influences the output: attention and max pooling mask pad positions, so the same review padded to 150 or 200 gives the same logits.
- A model that predicts only the positive class gives Negative F1 = 0.0 and valid other metrics, not an exception or NaN.
- A GloVe file with a malformed line or a wrong-dimension vector skips that line instead of crashing; words missing from GloVe get a small random vector, `<pad>` stays zeros.
- The same seed gives identical validation metrics on a tiny CPU run (reproducibility for question (b)).

---

## File Structure

- Create: `src/glove_bilstm.py`: all reusable logic (one file, about 250 lines)
- Create: `tests/test_glove_bilstm.py`: pytest tests with tiny fixtures (no real GloVe, no downloads)
- Create: `requirements.txt`, `.gitignore`: dependencies; ignore `data/glove/`, `results/`
- Create: `02_GloVe_BiLSTM.ipynb`: experiment driver
- Output (not committed): `data/glove/`, `results/bilstm/`

### Task 1: Environment, data loading, vocabulary, encoding

**Files:**
- Create: `requirements.txt` (torch, numpy, scikit-learn, matplotlib, pandas, pytest, jupyter, nbconvert), `.gitignore` (`data/glove/`, `results/`, `__pycache__/`, `.ipynb_checkpoints/`), `src/__init__.py`, `src/glove_bilstm.py`
- Test: `tests/test_glove_bilstm.py`

**Interfaces:**
- Produces: `tokenize(text: str) -> list[str]`; `build_vocab(texts: list[str], min_freq: int = 2) -> dict[str, int]` (contains `<pad>`:0, `<unk>`:1); `encode(text: str, vocab: dict[str, int], max_len: int) -> list[int]` (always length `max_len`); `load_split(path: str) -> tuple[list[str], list[int]]`.

- [ ] **Step 1: Write failing tests** in `tests/test_glove_bilstm.py`: `test_tokenize_keeps_negation_and_punct` (`tokenize("Not good!")==["not","good","!"]`); `test_build_vocab_specials_and_min_freq` (pad 0, unk 1, a word seen once is absent with `min_freq=2`); `test_encode_pads_and_truncates` (length equals `max_len` for short and long input); `test_encode_empty_and_all_oov` (all-`<unk>`/all-pad, length `max_len`, no exception).
- [ ] **Step 2: Run** `python -m pytest tests -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** the four functions in `src/glove_bilstm.py`; `load_split` reads the JSON list of records and returns texts and int labels (labels absent → raise `KeyError`, since the test split must not be loaded here).
- [ ] **Step 4: Run** `python -m pytest tests -v`. Expected: PASS. Install deps first with `python -m venv .venv && .venv/bin/pip install -r requirements.txt`.
- [ ] **Step 5: Commit** `git add requirements.txt .gitignore src tests && git commit -m "feat: tokenizer, vocab and encoding for GloVe BiLSTM"`

### Task 2: GloVe download and embedding matrix

**Files:**
- Modify: `src/glove_bilstm.py`
- Test: `tests/test_glove_bilstm.py`

**Interfaces:**
- Consumes: `build_vocab` output from Task 1.
- Produces: `download_glove(dest_dir: str = "data/glove") -> str` (returns path to `glove.6B.100d.txt`; skips if present); `load_glove(path: str, dim: int = 100) -> dict[str, np.ndarray]`; `build_embedding_matrix(vocab: dict[str, int], glove: dict[str, np.ndarray], dim: int = 100, seed: int = 42) -> np.ndarray` (shape `(len(vocab), dim)`, float32).

- [ ] **Step 1: Write failing tests** with a tiny temp GloVe file (dim 4): `test_load_glove_skips_malformed_and_wrong_dim` (a bad line and a 3-dim line are skipped); `test_embedding_matrix_rows` (known word equals its GloVe vector, pad row all zeros, an OOV word gets a non-zero small random row, shape `(len(vocab), dim)`); `test_embedding_matrix_deterministic` (same seed → equal arrays).
- [ ] **Step 2: Run** `python -m pytest tests -v -k "glove or embedding"`. Expected: FAIL.
- [ ] **Step 3: Implement** the three functions; `download_glove` uses `urllib.request` to fetch `glove.6B.zip`, extracts only `glove.6B.100d.txt`, deletes the zip; random rows ~ `N(0, 0.1)` from `np.random.default_rng(seed)`.
- [ ] **Step 4: Run** the tests. Expected: PASS. Then run `python -c "from src.glove_bilstm import download_glove; print(download_glove())"` once; Expected: prints the path (about 350 MB file on disk).
- [ ] **Step 5: Commit** `git commit -am "feat: GloVe loading and embedding matrix"` (add changed files).

### Task 3: BiLSTM model

**Files:**
- Modify: `src/glove_bilstm.py`
- Test: `tests/test_glove_bilstm.py`

**Interfaces:**
- Consumes: embedding matrix from Task 2.
- Produces: `class BiLSTMClassifier(nn.Module)` with `__init__(self, embedding_matrix: np.ndarray, hidden: int = 128, layers: int = 1, dropout: float = 0.4, pooling: str = "attention", bidirectional: bool = True, freeze_embeddings: bool = True)` and `forward(self, ids: Tensor[B, L]) -> Tensor[B, 2]` (logits). `pooling` in `{"attention", "max", "last"}`; pad id 0 is masked.

- [ ] **Step 1: Write failing tests**: `test_forward_shape` for all three pooling modes and `bidirectional` True/False (`(B, 2)`); `test_padding_does_not_change_logits` (same review padded to 150 vs 200 → `torch.allclose` in `eval()` mode, for attention and max); `test_frozen_embeddings_have_no_grad` (and `freeze_embeddings=False` has grad); `test_all_pad_row_no_nan` (a row of all zeros gives finite logits).
- [ ] **Step 2: Run** `python -m pytest tests -v -k model`. Expected: FAIL.
- [ ] **Step 3: Implement** `BiLSTMClassifier`. Use `pack_padded_sequence` with lengths clamped to at least 1; attention is a learned vector scoring each position with a masked softmax; `last` uses the final real position of each direction.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -am "feat: BiLSTM classifier with attention/max/last pooling"`.

### Task 4: Metrics and training loop

**Files:**
- Modify: `src/glove_bilstm.py`
- Test: `tests/test_glove_bilstm.py`

**Interfaces:**
- Consumes: `BiLSTMClassifier` from Task 3; encoded id lists from Task 1.
- Produces: `compute_metrics(y_true, y_pred, y_prob) -> dict` with keys `accuracy, precision, recall, f1, neg_f1, pr_auc, confusion_matrix` (confusion as nested list `[[tn, fp], [fn, tp]]`); `train_model(model, train_ids, train_y, val_ids, val_y, *, lr=1e-3, batch_size=64, max_epochs=15, patience=3, clip=1.0, class_weighted=False, seed=42, device=None) -> dict` returning `{"history": [{"epoch", "train_loss", "val_loss", "val_neg_f1", "val_acc"}...], "best_epoch": int, "train_seconds": float, "metrics": dict}` with the model restored to its best-validation-Negative-F1 weights; `predict_proba(model, ids, batch_size=256, device=None) -> np.ndarray` (positive-class probability); `set_seed(seed: int) -> None`.

- [ ] **Step 1: Write failing tests**: `test_metrics_all_positive_prediction` (`neg_f1 == 0.0`, no exception/NaN, `accuracy` equals positive share); `test_metrics_perfect` (all 1.0, confusion `[[n0,0],[0,n1]]`); `test_train_model_reduces_loss_and_is_deterministic` (tiny synthetic separable data, CPU: train loss at last epoch below first, and two runs with the same seed give equal `metrics["accuracy"]` and `history`); `test_early_stopping_restores_best` (`best_epoch` ≤ epochs run, final model's val Negative F1 equals the recorded best).
- [ ] **Step 2: Run** `python -m pytest tests -v -k "metrics or train"`. Expected: FAIL.
- [ ] **Step 3: Implement.** `compute_metrics` uses scikit-learn with `zero_division=0` and `neg_f1 = f1_score(..., pos_label=0)`; `pr_auc` is average precision of the positive class. `class_weighted=True` uses inverse-frequency class weights from the training labels. Seed Python, numpy and torch; choose `mps`/`cuda`/`cpu` automatically.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -am "feat: metrics and training loop with early stopping"`.

### Task 5: Experiment notebook and results

**Files:**
- Create: `02_GloVe_BiLSTM.ipynb`
- Output: `results/bilstm/experiments.csv`, `results/bilstm/best_final.json`, `results/bilstm/curves.png`, `results/bilstm/confusion.png`, `results/bilstm/pr_curve.png`, `results/bilstm/errors.csv`

**Interfaces:**
- Consumes: every function listed in Tasks 1–4.
- Produces: `results/bilstm/experiments.csv` with columns `name, variable, value, best_epoch, accuracy, precision, recall, f1, neg_f1, pr_auc, train_seconds, params`.

- [ ] **Step 1: Write the notebook** with sections: setup and seed; load splits and build vocab/embeddings from train only; baseline config dict (the Global Constraints model); `run(config)` helper; experiment loop over the one-variable list (each run seed 42, results appended to `experiments.csv`, skipping runs already in the CSV); pick the best by validation Negative F1 (ties broken by PR-AUC); re-run it with seeds 42/43/44 and write mean ± std to `best_final.json`; plot training curves, confusion matrix and PR curve for the best seed-42 run; write up to 20 misclassified validation reviews (10 per error type, lowest-confidence first) to `errors.csv`; print the comparison table.
- [ ] **Step 2: Execute** `.venv/bin/jupyter nbconvert --to notebook --execute --inplace 02_GloVe_BiLSTM.ipynb`. Expected: completes without error; `experiments.csv` has one row per experiment; all plots exist.
- [ ] **Step 3: Sanity-check results.** Expected: every `accuracy` and `neg_f1` is between 0 and 1; the best run's validation accuracy is above the 85.4% always-positive baseline or the notebook prints a clear note explaining why not; no run reads `test_split.json` (`grep -c test_split 02_GloVe_BiLSTM.ipynb` prints 0).
- [ ] **Step 4: Commit** `git add 02_GloVe_BiLSTM.ipynb && git commit -m "feat: GloVe BiLSTM experiments on validation split"`. Results and GloVe stay untracked.

### Task 6: Final review and handoff notes

**Files:**
- Modify: `README.md` (Project Status line only), `CLAUDE.md` (add `02_GloVe_BiLSTM.ipynb`, `src/`, test command `python -m pytest tests -v`)

- [ ] **Step 1: Run** `python -m pytest tests -v`. Expected: all pass.
- [ ] **Step 2: Request review** of the whole branch diff against the Global Constraints and Review Focus, and fix what the reviewer confirms is wrong.
- [ ] **Step 3: Update** the README status and CLAUDE.md; **Commit** `git commit -am "docs: document GloVe BiLSTM notebook"`.
