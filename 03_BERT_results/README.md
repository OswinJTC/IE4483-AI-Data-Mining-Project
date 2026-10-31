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
