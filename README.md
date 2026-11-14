# Water Quality Prediction

**Project 18 · Water Quality Prediction via Gradient Boosting and Deep Neural Networks**

This project estimates the continuous, 0–100 `CCME_Values` water-quality index from eight water measurements. It has two independent parts:

| Part | Folder | Model | Test RMSE (2021–2023) |
|---|---|---|---:|
| **A · Applied Machine Learning** | [`AML/`](AML) | XGBoost (vs mean and Ridge baselines), 300k-row sample | 0.426 |
| **B · Deep Learning** | [`DL/`](DL) | PyTorch MLP trained on the whole dataset | **0.136** |

**Target interpretation:** the released index is calculated from the same eight measurements used as predictors. Our formula audit reconstructs every label within `1e-7`. The results therefore measure **index emulation**, not independent environmental outcomes or drinking-water safety.

## Team

| Team member | Enrollment number |
|---|---|
| Dhruv Kumar | 2401010152 |
| Ritesh | 2041010383 |
| Yash Kumar | 2401010512 |
| Parul | 2401010321 |
| Punit | 2401010359 |

## Repository structure

```text
Water_Quality_Prediction/
├── Datasets/
│   └── Water_Quality_Dataset.zip        # shared source data (both parts)
├── AML/                                 # Part A · XGBoost
│   ├── Notebooks/  01_eda · 02_preprocessing · 03_xgboost_model
│   ├── Models/  Results/  figures/
│   ├── Reports/                         # project report, literature review, original 12-slide deck
│   └── README.md
├── DL/                                  # Part B · neural network
│   ├── Notebooks/  01_data_preparation · 02_mlp_training · 03_final_model
│   ├── Models/  Results/  figures/
│   ├── Reports/                         # 8-slide DL presentation (PPTX + PDF)
│   └── README.md
├── .gitattributes
├── .gitignore
└── README.md
```

Each part is self-contained. It reads the shared CSV from `Datasets/` and writes only inside its own folder. The one link between them: the DL results table shows the AML XGBoost score as a reference when `AML/Results/metrics.csv` exists. Generated caches go in `Datasets/_working/`, `AML/_working/` and `DL/_working/`. Git ignores them, and you can delete them after use.

## Setup

Use **Python 3.12**. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.3.4 pandas==2.3.3 scikit-learn==1.7.2 xgboost==3.1.2 torch==2.9.1 matplotlib==3.11.2 joblib==1.6.0 ipykernel notebook
python -m ipykernel install --user --name water-quality --display-name "Water Quality (Python 3.12)"
python -m notebook
```

On Windows, activate with `.venv\Scripts\activate`. Then run each part's notebooks in numeric order. See [`AML/README.md`](AML/README.md) and [`DL/README.md`](DL/README.md) for details. Both parts ship with trained models: set `RETRAIN = False` in the last notebook to use them without retraining.

**Verification:** all six notebooks were executed from scratch in order, without errors. The AML part reproduces the original XGBoost test predictions exactly. Each part's final notebook reloads its saved model files, verifies their checksums and re-predicts held-out examples.

## Data

[Karim et al., Figshare dataset, version 2](https://doi.org/10.6084/m9.figshare.27800394.v2): **2,827,977 rows**, 14 columns, five countries, 1940–2023, redistributed with attribution under **CC BY 4.0**. Source CSV SHA-256: `81ced159f465aa5ab004b77130e5f7147c698876ddbf2f97807e31c9d0e8d714`.

The inputs are ammonia, BOD, dissolved oxygen, orthophosphate, pH, temperature, nitrogen and nitrate, plus one missing/invalid flag for each. Exact duplicates (9,971 rows) are removed. Both parts use the same chronological split: training through 2018, validation 2019, calibration 2020 and test 2021–2023 (202,886 rows). Test rows never select a model.

## Results

| Model | Training rows | Test MAE ↓ | Test RMSE ↓ | Test R² ↑ |
|---|---:|---:|---:|---:|
| Mean baseline (AML) | 300,000 | 11.7909 | 13.4584 | −0.4141 |
| Ridge (AML) | 300,000 | 4.3442 | 5.5675 | 0.7580 |
| XGBoost (AML) | 300,000 | 0.1099 | 0.4262 | 0.99858 |
| MLP, original setup (DL) | 300,000 | 0.4649 | 0.9223 | 0.99336 |
| **MLP, full data (DL)** | 2,478,761 | **0.0279** | **0.1357** | **0.99986** |

The original study found that XGBoost beat a small MLP trained on a 300k sample. In the DL part, the same 128-64 network trained on all 2.48 M rows, without dropout and with a one-cycle learning-rate schedule, lowers RMSE 6.8× and overtakes XGBoost. XGBoost was not retrained on the full data, so the two parts do not compare at equal data size. Orthophosphate and ammonia dominate the feature importance for both models, because both enter the source formula directly. Interval coverage is close to 90% overall but much lower in the rare Marginal and Poor bands. Computing the source formula is preferable whenever all eight inputs are available.

Methods: [XGBoost](https://doi.org/10.1145/2939672.2939785), [Deep Learning, chapter 6](https://www.deeplearningbook.org/contents/mlp.html), [one-cycle schedule](https://arxiv.org/abs/1708.07120), [dropout](https://jmlr.org/papers/v15/srivastava14a.html), [residual/conformal intervals](https://arxiv.org/abs/2107.07511). Official standard for comparison: [CCME manual](https://ccme.ca/en/res/wqimanualen.pdf).
