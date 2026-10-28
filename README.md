# IE4483 — Sentiment Analysis of Product Reviews

## 1. Project Setup

We compare three sentiment classification approaches:

- **Traditional ML:** TF-IDF + Classifier
- **RNN:** Word Embeddings + RNN
- **BERT:** Pretrained Transformer

`00_Data_Preparation.ipynb` cleans the raw data and generates a shared **85/15 stratified train-validation split (seed = 42)**. All models must load the same files from `data/processed/` and must not re-clean or re-split the data.

## 2. Data

There is no `test.json`. `data/raw/train.json` is the only labeled data and is split into train and validation. The test set is the 1,851 unlabeled reviews in the `data/raw/submission.csv` template, and the final predictions go into a `submission.csv` with the same rows and order.

| Path | Contents |
|---|---|
| `data/raw/train.json` | Labeled reviews (`reviews`, `sentiments`: 1 = positive, 0 = negative) |
| `data/raw/submission.csv` | Test reviews with placeholder predictions (`reviews`, `predictions`) |
| `data/processed/train_split.json` | Training split (`reviews`, `sentiments`) |
| `data/processed/val_split.json` | Validation split (`reviews`, `sentiments`) |
| `data/processed/test_split.json` | Cleaned test reviews, original order, no labels |
| `data/processed/prep_summary.json` | Row counts and class ratios for the report |

### Data preparation

The notebook runs locally and needs `pandas` and `scikit-learn`. Run it from the repo root so the relative `data/` paths resolve:

```
jupyter nbconvert --to notebook --execute --inplace 00_Data_Preparation.ipynb
```

Steps:

1. **Clean (shared by all models):** unescape HTML entities, strip HTML tags, normalize unicode and whitespace. Casing and punctuation are kept so BERT is unaffected; lowercasing and tokenization stay in each model notebook.
2. **Filter (train only):** drop empty, conflicting-label and duplicate reviews before splitting, so no review can appear in both train and validation. The test set keeps every row and its order.
3. **Split:** stratified 85/15, seed 42, so the negative/positive ratio matches in train and validation. The notebook asserts the gap is under 0.1 percentage points.

| | Rows | Negative | Positive |
|---|---|---|---|
| Raw train | 7,401 | 1,082 | 6,319 |
| After filtering (219 duplicates removed) | 7,182 | 14.59% | 85.41% |
| Train | 6,104 | 14.60% | 85.40% |
| Validation | 1,078 | 14.56% | 85.44% |
| Test | 1,851 | – | – |

Re-running the notebook reproduces the same files byte for byte.

## 3. Evaluation Standard

To ensure a fair comparison, all three models should report consistent metrics.

| Evaluation | Traditional ML | RNN | BERT |
|---|---|---|---|
| Accuracy | ✓ | ✓ | ✓ |
| Precision / Recall | ✓ | ✓ | ✓ |
| F1 / Negative F1 | ✓ | ✓ | ✓ |
| Confusion Matrix | ✓ | ✓ | ✓ |
| PR Curve / PR-AUC | ✓* | ✓ | ✓ |
| Error Analysis | ✓ | ✓ | ✓ |
| Training Time | ✓ | ✓ | ✓ |
| Training Curves | Optional | ✓ | ✓ |

*PR-AUC requires prediction scores or probabilities.*

## 4. Results & Reporting

- Evaluate all models on the **same validation set**. The test set has no labels, so correct/incorrect analysis uses the validation set.
- Use `1 = Positive`, `0 = Negative`.
- Report **Negative F1** because the dataset is class-imbalanced (about 85% positive; an always-positive model scores about 85% accuracy).
- Compare classification performance, computational cost, and representative prediction errors.
- Present the best configuration of each model in one final comparison table.

## 5. Project Status

**Experiments in progress.** Data preparation is done, and the GloVe + BiLSTM model (`02_GloVe_BiLSTM.ipynb`) has been run on the validation set: best configuration (fine-tuned GloVe, attention pooling) reaches about 91.3% accuracy and 0.70 Negative F1 over 3 seeds. These validation scores were also used for model selection, so treat them as optimistic. Final metrics and visualizations will be updated after model evaluation is completed.

The best-performing model will generate the final `submission.csv`.
