# BERT Experiment Results

Start with [BERT Experiment Analysis Summary](BERT_Experiment_Analysis_Summary.pdf) for the full experiment path, findings, selected Run 10, and guidance for the group report. You do not need to read every notebook.

## Folder structure

| Folder | Purpose |
|---|---|
| `BERT_Stage1_Epochs/` | Compare training lengths. |
| `BERT_Stage2_Learning_Rate/` | Tune learning rate. |
| `BERT_Stage3_Batch_Size/` | Compare batch sizes. |
| `BERT_Stage4_Dropout/` | Tune dropout. |
| `BERT_Stage5_Weight_Decay/` | Tune weight decay. |
| `BERT_Stage6_Freezing_and_Early_Stopping/` | Compare layer freezing and early stopping; contains selected Run 10. |
| `BERT_Stage7_Seed_Stability/` | Check the selected settings across seeds. |
| `BERT_Stage8_0_Search_Plan/` | Shared `stage8_search_plan.csv` defining all eight search candidates. |
| `BERT_Stage8_1_Grid_Search/`, `BERT_Stage8_2_Grid_Search/` | Four grid-search candidates in total. |
| `BERT_Stage8_3_Random_Search/`, `BERT_Stage8_4_Random_Search/` | Four random-search candidates in total. |

Each experiment stage contains:

- **`experiment.ipynb`**: code for that stage.
- **`stage*_comparison.csv`** or **`stage8_results.csv`**: comparison of run results. Stage 8 also includes a copy of the search plan.
- **`Run_*/`**: `config.json` (parameters), `metrics.json` (scores), `training_history.csv` (training logs), `validation_predictions.csv` (error analysis), `submission.csv` (test predictions), and `figures/` (plots). Some carried-forward reference runs contain only `metrics.json`.

Model weights (`best_model/`), training checkpoints, and local temporary files are excluded via `.gitignore` because of their size. This repository retains the code and experiment evidence, not trained models.

Run 10's test predictions are in [submission.csv](BERT_Stage6_Freezing_and_Early_Stopping/Run_10_EarlyStopping/submission.csv). Read the [PDF summary](BERT_Experiment_Analysis_Summary.pdf) for why it was selected and the limitations of the results.

## Extra experiment: GPT-OSS 120B pseudo-labels

Start with the [test evaluation](BERT_Extra_GPT_OSS_Evaluation/test/test_evaluation.txt): Run 10 scored **95.30% accuracy** and **0.8455 Negative F1** against GPT-OSS 120B pseudo-labels on 1,851 test reviews. The report includes the confusion matrix. These are scores against model-generated labels.

- **Validation check:** [summary](BERT_Extra_GPT_OSS_Evaluation/validation/validation_summary.txt) and [1,078 predictions](BERT_Extra_GPT_OSS_Evaluation/validation/validation_predictions.csv). GPT-OSS 120B scored **96.01% accuracy** and **0.8732 Negative F1** against validation ground truth.
- **Test labels:** [original pseudo-labels with row IDs](BERT_Extra_GPT_OSS_Evaluation/test/pseudo_labels.csv) and [reference CSV in submission format](BERT_Extra_GPT_OSS_Evaluation/test/submission_actual_result.csv). The latter is for this comparison; the course submission is the selected model's own prediction CSV.
- **[Scripts](BERT_Extra_GPT_OSS_Evaluation/scripts/):** `label_validation.py` and `label_test.py` preserve the labeling workflow and leave completed outputs in place. Credentials live in the ignored root `.groq_keys` file or environment variables.

Recompute the test comparison from the repository root (no API key or model weights required):

```bash
python3 03_BERT_results/BERT_Extra_GPT_OSS_Evaluation/scripts/compare_test_predictions.py
```
