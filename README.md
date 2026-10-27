# IE4483 — Sentiment Analysis of Product Reviews

## 1. Project Setup

We compare three sentiment classification approaches:

- **Traditional ML:** TF-IDF + Classifier
- **RNN:** Word Embeddings + RNN
- **BERT:** Pretrained Transformer

`00_Data_Preparation.ipynb` generates a shared **85/15 stratified train-validation split (seed = 42)**. All models use the same datasets from Google Drive.

## 2. Evaluation Standard

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

## 3. Results & Reporting

- Evaluate all models on the **same validation set**.
- Use `1 = Positive`, `0 = Negative`.
- Report **Negative F1** because the dataset is class-imbalanced.
- Compare classification performance, computational cost, and representative prediction errors.
- Present the best configuration of each model in one final comparison table.

## 4. Project Status

**Experiments in progress.** Final metrics and visualizations will be updated after model evaluation is completed.

The best-performing model will generate the final `submission.csv`.

## 5. Accessing Shared Datasets (Google Drive)

All datasets have already been prepared and shared via Google Drive.

**Shared Folder:** [IE4483 Materials](https://drive.google.com/drive/folders/10lKm8HjEw9RLhBaLM7KV1F2xK_0zZi-J?usp=sharing)

### Setup Instructions

1. Open the shared Google Drive link above.
2. Right-click **IE4483 materials → Organize → Add shortcut**.
3. Save the shortcut to **My Drive → Colab Notebooks**.
4. **This step is mandatory to use the predefined `SHARED_DIR` path without modification:**

```python
SHARED_DIR = "/content/drive/MyDrive/Colab Notebooks/IE4483 materials/processed_data"
```