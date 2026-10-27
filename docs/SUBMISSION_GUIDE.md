# Submission guide

## Files to present

| Item | Location |
| --- | --- |
| Public repository | https://github.com/hypnoastic/Water_Quality_Prediction |
| Main project report | `submission/Water_Quality_Project_Report.pdf` |
| Literature review | `submission/Water_Quality_Literature_Review.pdf` |
| PowerPoint | `presentation/Water_Quality_Prediction.pptx` |
| Presentation PDF | `presentation/Water_Quality_Prediction.pdf` |
| EDA notebook | `notebooks/02_eda.ipynb` |
| Preprocessing notebook | `notebooks/03_preprocessing.ipynb` |
| Models and inference notebook | `notebooks/04_model_application.ipynb` |
| Quick results walkthrough | `notebooks/01_results.ipynb` |
| Final technical verification | `submission/VERIFICATION.md` |
| Team metadata | `submission/team.json` |

Team names, enrollments, team ID, section and institution are intentionally dummy
values, as requested. Replace `submission/team.json`, then rebuild the PDFs and PPT.
The rest of the package uses actual data, measured results and trained model files.
The compact `submission/Water_Quality_Submission.zip` includes code, models and all
documents. The two larger original-data archives stay separate in the public repo.
The ZIP still supports inference immediately and can download training data with
`python scripts/download_data.py`.

## Five-minute demonstration

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install -e . --no-deps --no-build-isolation
streamlit run app.py
```

1. Explain the target and the formula audit first.
2. In the demo, keep the default measurements and click Predict.
3. Switch between XGBoost, MLP and Blend to compare real saved-model predictions.
4. Show the chronological results and validation feature importance.
5. Explain that this is source-index emulation, not a water-safety decision.

The demo and all submitted results work without downloading the 320 MB raw CSV.
Retraining and the EDA/preprocessing notebooks need that file:

```bash
python scripts/download_data.py
python -m water_quality train --config configs/default.json
```

## Suggested presentation order

Problem, dataset, target audit, preprocessing and time splits, model design, results,
feature explanations, interval coverage, limitations, conclusion and references.
Speaker notes in the PowerPoint provide explanations and source links.

## Scope to defend in the viva

The project meets the assignment as a comparison on a documented continuous index.
Its scientific contribution is the audit and a fair model comparison. Because the
target is derived from the same inputs, the results do not demonstrate independent
environmental prediction. Temporal separation does not remove this structural
dependence or reverse upstream imputation. The neural model is a useful comparator
even though XGBoost wins individually. The blend has a small RMSE advantage and
worse MAE than XGBoost. No significance claim is justified by one seed.

## Rebuilding submission documents

```bash
python scripts/build_submission.py
python scripts/create_notebooks.py
python scripts/verify_submission.py
```

The presentation builder uses PptxGenJS. Install its exact version from
`presentation/package-lock.json`, then run the supplied script:

```bash
cd presentation
npm ci
node build_deck.mjs
```

For an optional PDF export with LibreOffice:

```bash
soffice --headless --convert-to pdf --outdir presentation presentation/Water_Quality_Prediction.pptx
```
