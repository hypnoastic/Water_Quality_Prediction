# Part A · Applied Machine Learning — XGBoost

Gradient-boosted decision trees (XGBoost) estimate the continuous, 0–100 `CCME_Values` index from eight water measurements. A training-mean baseline and Ridge regression are the reference points. The neural-network part is in [`../DL`](../DL).

## Folder contents

| Path | What it holds |
|---|---|
| `Notebooks/01_eda.ipynb` | Verifies the source CSV, inspects units, missing values and duplicates, audits the target formula and exports the exploratory charts |
| `Notebooks/02_preprocessing.ipynb` | Cleans the records, splits them by date, fits training-only transforms and writes the local cache in `_working/` |
| `Notebooks/03_xgboost_model.ipynb` | Trains XGBoost (depth 4 vs 6) and the baselines, evaluates on 2021–2023, builds 90% intervals, runs permutation importance and TreeSHAP, saves the model and reloads it to predict |
| `Models/` | `xgboost.json`, `preprocessing.joblib`, `baselines.joblib`, `metadata.json`, `checksums.json` |
| `Results/` | Test predictions, metrics, subgroup scores, importance tables, training history, data audit, run manifest, prediction example |
| `figures/` | Data overview, target and time coverage, correlations, temporal split, model comparison, actual vs predicted, feature importance, learning curves, interval coverage |
| `Reports/` | The project report, literature review and original 12-slide presentation (PPTX and PDF), covering both parts of the original study |

## Method

* **Data:** 2,827,977 rows, minus 9,971 exact duplicates. Inputs are the eight measurements plus eight missing/invalid flags (16 columns). Median imputation is fitted on training rows only, and the trees use unscaled inputs.
* **Split by date:** training 1940–2018 (a seed-42 sample of 300,000 rows), validation 2019, calibration 2020 and test 2021–2023 (all 202,886 rows).
* **XGBoost:** histogram trees with learning rate 0.05, up to 1,000 rounds and early stopping after 40 rounds without improvement. Depth 6 is selected on validation, and its best round is 803.

## Results (test 2021–2023)

| Model | MAE ↓ | RMSE ↓ | R² ↑ | 90% interval coverage |
|---|---:|---:|---:|---:|
| Mean baseline | 11.7909 | 13.4584 | −0.4141 | 94.5% |
| Ridge | 4.3442 | 5.5675 | 0.7580 | 84.2% |
| **XGBoost** | **0.1099** | **0.4262** | **0.9986** | 88.4% |

Orthophosphate and ammonia dominate both the permutation importance and TreeSHAP, because both enter the source formula directly. The index is calculated from the same inputs, so these scores measure **index emulation**, not independent water-quality prediction.

## Run

From the repository root, after the setup in the [main README](../README.md), run `01 → 02 → 03`. To use the saved model without retraining, open `03_xgboost_model.ipynb`, set `RETRAIN = False` and run all cells.
