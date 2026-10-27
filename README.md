# Water Quality Prediction

**Project 18: Gradient Boosting and Deep Neural Networks**

A reproducible comparison of XGBoost and a regularized PyTorch MLP using the
published `CCME_Values` index. Includes trained models, a local demo, executed
notebooks, research results and the final submission documents.

[![Verification](https://github.com/hypnoastic/Water_Quality_Prediction/actions/workflows/verify.yml/badge.svg)](https://github.com/hypnoastic/Water_Quality_Prediction/actions/workflows/verify.yml)

| Read or present | File |
| --- | --- |
| Project report | [10-page PDF](submission/Water_Quality_Project_Report.pdf) |
| Literature review | [4-page PDF](submission/Water_Quality_Literature_Review.pdf) |
| Presentation | [Editable PowerPoint](presentation/Water_Quality_Prediction.pptx) · [PDF](presentation/Water_Quality_Prediction.pdf) |
| Detailed results | [Technical report](reports/report.md) |
| Theory and lessons | [learning.md](learning.md) |
| Verification | [Submission checks](submission/VERIFICATION.md) |

![Model comparison](reports/figures/model_comparison.png)

## Target and scope

We predict one continuous score on a 0–100 scale from ammonia, BOD, dissolved
oxygen, orthophosphate, pH, temperature, nitrogen and nitrate. The supplied dataset
contains 2,827,977 rows from five countries across 1940–2023. Turbidity, conductivity
and dissolved-solids measurements are not available.

**The target is calculated from the same eight inputs.** Our audit reproduces every
released label within `1e-7`, including the source denominator of 12. This project
therefore evaluates **index emulation**. It does not establish independent water-quality
prediction, forecasting or drinking-water safety. The source's row-level adaptation
differs from official CCME site-period WQI. See [the data card](docs/DATA.md).

## Run the trained project

From the repository root, using Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install -e . --no-deps --no-build-isolation

# Interactive demo: one sample or a CSV, using the saved models
streamlit run app.py
```

The demo and CLI work without downloading the raw dataset. For CLI inference:

```bash
python -m water_quality predict --input examples/measurements.csv
```

Use `--model XGBoost`, `--model MLP` or `--model Blend` to select a model.
Inputs must retain the example's original headers and units. The default blend
uses 90% XGBoost and 10% MLP, selected by validation RMSE. Predictions are written
to `reports/example_predictions.csv`, an ignored runtime output.

## Reproduce the research

```bash
# Download the exact combined CSV and verify its SHA-256
python scripts/download_data.py

# Audit, train, select, calibrate, evaluate and explain
python -m water_quality train --config configs/default.json

# Rebuild figures and the Markdown report without retraining
python -m water_quality report

# Check implementation and submission evidence
python -m pytest -q
ruff check src tests scripts app.py
python scripts/verify_submission.py
```

`--skip-raw` lets the submission verifier run without an extracted dataset.
The combined CSV stays local and is excluded from Git. Download details are in
[data/README.md](data/README.md). A shallow clone avoids downloading obsolete files
from earlier repository history:

```bash
git clone --depth 1 https://github.com/hypnoastic/Water_Quality_Prediction.git
```

## Methods and measured results

All models share a seed-42 sample of 300,000 historical training rows. Training ends
in 2018, validation uses 2019, residual calibration uses 2020 and the full test
period covers 2021–2023. Median imputation fits on training only. Fixed quality flags
retain information about missing/invalid readings. The MLP and Ridge additionally
use `asinh` and standard scaling. Pollution extremes are preserved.

| Model | Purpose | Test MAE | Test RMSE | Test R² |
| --- | --- | ---: | ---: | ---: |
| Mean baseline | Constant reference | 11.7909 | 13.4584 | -0.41409 |
| Ridge | Additive reference | 4.3442 | 5.5676 | 0.75799 |
| XGBoost | Strong threshold and interaction model | 0.1099 | 0.4262 | 0.99858 |
| MLP | Learned nonlinear combinations | 0.4466 | 0.9161 | 0.99345 |
| Blend | Validation-selected combination | 0.1267 | 0.4222 | 0.99861 |

These scores use all **202,886 test rows**. XGBoost has the lowest MAE. The blend
slightly improves RMSE, with no significance claim from one seed. Its nominal 90%
intervals cover 89.3% overall, but only 45.2% of Marginal rows. Native TreeSHAP and
validation permutation importance describe model reliance, not causal effects.

The source was already imputed upstream. Unit questions remain, countries are
unequally represented and historical sites may recur. These limits remain explicit
in the reports. An independent repeat of training reproduced the metrics exactly.

## Files kept and their purpose

| Location | Purpose |
| --- | --- |
| `app.py` | Streamlit form and CSV demo using saved models |
| `src/water_quality/` | Data rules, training, evaluation, explanations, inference and CLI |
| `configs/default.json` | Reproducible experiment settings |
| `artifacts/` | Selected tree/MLP, fitted preprocessing, model metadata and integrity hashes |
| `examples/` | Input sample and expected predictions for demonstrations and reload tests |
| `data/` | Versioned download instructions and source hashes; one training CSV locally |
| `notebooks/` | Executed EDA, preprocessing and model-application notebooks |
| `notebooks/reference/` | Original harmonization notebook documenting how labels were calculated |
| `reports/` | Metrics, test predictions, figures, candidate history, audit and experiment manifests |
| `submission/` | Final project/literature PDFs, editable team details and one verification record |
| `presentation/` | Final PPT/PDF and the source needed to rebuild the editable deck |
| `scripts/` | Download data, rebuild PDFs, repair generated PPT packaging and verify evidence |
| `tests/` and `.github/workflows/` | Local tests and automated GitHub verification |
| `pyproject.toml`, `requirements.lock.txt` | Package configuration and exact dependencies |
| `learning.md`, `docs/DATA.md` | Concise theory and detailed target/provenance documentation |
| `LICENSE`, `CITATION.cff`, Git configuration files | Usage rights, citation and repository hygiene |

Read the notebooks in order: [EDA](notebooks/01_eda.ipynb),
[preprocessing](notebooks/02_preprocessing.ipynb), then
[model application](notebooks/03_model_application.ipynb). The first two need the CSV.
The third reads delivered results and performs saved-model inference.

## Team details and document rebuilds

Team identities intentionally remain dummy values. Edit `submission/team.json`,
then run `python scripts/build_submission.py` to rebuild the two report PDFs.
For the presentation, run `npm ci` followed by `node build_deck.mjs` inside
`presentation/`. The builder reads the actual results directly from `reports/`.
The final PDF can be exported from PowerPoint or LibreOffice after editing the PPT.

## Attribution

Dataset: Karim et al., [Figshare v2](https://doi.org/10.6084/m9.figshare.27800394.v2),
**CC BY 4.0**. The original harmonization notebook remains unchanged. New code and
analysis use the [MIT license](LICENSE). Presentation photo credit is in
[presentation/assets/README.md](presentation/assets/README.md).

The user-supplied [flight-delay repository](https://github.com/abhi-7-7/Flight_Delay_Prediction)
and [Canva presentation](https://canva.link/jhwgwqkgeon4gbs) informed the submission
organization and presentation structure. No flight data or results enter this project.
