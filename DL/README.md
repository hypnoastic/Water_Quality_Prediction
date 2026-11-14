# Part B · Deep Learning — PyTorch MLP

A feedforward neural network (multilayer perceptron) estimates the continuous, 0–100 `CCME_Values` index from eight water measurements. It is trained on the **whole** training history (2.48 million rows) and then refitted once on every year. The gradient-boosting part is in [`../AML`](../AML).

## Folder contents

| Path | What it holds |
|---|---|
| `Notebooks/01_data_preparation.ipynb` | Verifies and cleans all 2.83 M rows, splits them by date and fits the neural input pipeline on training years (quality flags → median imputation → `asinh` → standardization) |
| `Notebooks/02_mlp_training.ipynb` | Reproduces the original 300k-sample network, then trains four candidate networks on all training rows and selects one on 2019 validation |
| `Notebooks/03_final_model.ipynb` | Scores the selected network once on 2021–2023, builds 90% intervals, runs permutation importance, refits on all 1940–2023 rows, saves and reloads the networks |
| `Models/mlp_evaluation.pt` | Selected network trained on all 1940–2018 rows. All test scores below come from this model |
| `Models/mlp_final_all_years.pt` | Same design refitted on all 2,818,006 cleaned rows (1940–2023), for deployment. It has no held-out test score of its own |
| `Models/preprocessing.joblib`, `metadata.json`, `checksums.json` | Fitted input pipelines (`evaluation` and `final`), architecture and training settings, and file checksums |
| `Results/` | Architecture search, test predictions, metrics, subgroup scores, permutation importance, training histories, run manifest and a prediction example |
| `figures/` | Input scaling, temporal split, learning curves, model comparison, actual vs predicted, feature importance, performance by band |
| `Reports/Water_Quality_DL_Presentation.pptx` (+ `.pdf`) | The 8-slide presentation for this part |

## Selected network

**16 inputs → 128 → 64 → 1** with ReLU and no dropout (10,497 parameters). It is trained with AdamW (weight decay 1e-4) on MSE of the index divided by 100, using batch size 2,048, 60 epochs and a one-cycle learning rate that peaks at 0.003. The checkpoint is the best epoch on 2019 validation.

| Candidate (validation 2019) | Training rows | RMSE ↓ |
|---|---:|---:|
| Original: 128-64, dropout 0.1, fixed LR | 300,000 | 0.934 |
| A: 128-64, dropout 0.1 | 2,478,761 | 0.464 |
| **B: 128-64, no dropout** | 2,478,761 | **0.155** |
| C: 256-128, no dropout | 2,478,761 | 0.255 |
| D: 256-128-64, no dropout | 2,478,761 | 0.652 |

Three findings came out of the search. Training on all rows gives the largest gain. Dropout hurts, because the label has no noise. With the same recipe, the larger networks trained less stably.

## Results (test 2021–2023, 202,886 rows)

| Model | MAE ↓ | RMSE ↓ | R² ↑ | 90% coverage |
|---|---:|---:|---:|---:|
| MLP, 300k sample (original setup) | 0.4649 | 0.9223 | 0.99336 | 90.6% |
| **MLP, full data (this part)** | **0.0279** | **0.1357** | **0.99986** | 89.7% |
| XGBoost, 300k sample (from `AML/`, reference) | 0.1099 | 0.4262 | 0.99858 | 88.4% |

Trained on all the data, the MLP has 6.8× lower RMSE than the original small-sample MLP. It also beats the XGBoost reference, but XGBoost was trained on only the 300k sample, so part of the gap comes from data size. The interval radius is ±0.05 index points. Coverage is 98% in the Excellent band but only 35% in the rare Poor band. Orthophosphate and ammonia dominate permutation importance.

These numbers measure **index emulation**. The label is a formula of the same eight inputs, so this is not a drinking-water safety prediction. Results come from a single seed.

## Run

From the repository root, after the setup in the [main README](../README.md), run `01 → 02 → 03`. Notebook 02 takes about 6 minutes on a laptop CPU, and 03 about 1 minute. To use the saved networks without retraining, open `03_final_model.ipynb`, set `RETRAIN = False` and run all cells. `predict_measurements(frame)` accepts a DataFrame with the eight original column headers and units.
