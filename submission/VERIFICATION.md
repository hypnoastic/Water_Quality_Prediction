# Final submission verification

Status: **PASS**. 28 checks completed.
Verified UTC: 2026-10-27T03:36:52.024930+00:00

Apt for the brief as a transparent index-emulation comparison; independent environmental prediction is not established.

Team identities remain editable dummy values at the user's request.

| Check | Result | Evidence |
| --- | --- | --- |
| Submission artifacts | PASS | 15 required files |
| Trained artifact hashes | PASS | 4 saved artifact hashes match |
| Data accounting | PASS | 2,818,006 retained rows across disjoint dated partitions |
| One continuous target | PASS | CCME_Values regression, eight measurement inputs plus flags |
| Formula disclosure | PASS | Every original label reconstructs; derived-target limitation is explicit |
| Chronological separation | PASS | Training, validation, calibration and testing occupy separate ordered periods |
| Shared training sample | PASS | 300,000 seed-42 historical rows train every candidate |
| Full test predictions | PASS | 202,886 test rows with unique traceable source IDs |
| Recomputed metrics: Mean baseline | PASS | MAE 11.790940; RMSE 13.458382; R2 -0.414095 |
| Recomputed metrics: Ridge | PASS | MAE 4.344161; RMSE 5.567582; R2 0.757994 |
| Recomputed metrics: XGBoost | PASS | MAE 0.109925; RMSE 0.426203; R2 0.998582 |
| Recomputed metrics: MLP | PASS | MAE 0.446613; RMSE 0.916082; R2 0.993448 |
| Recomputed metrics: Blend | PASS | MAE 0.126738; RMSE 0.422248; R2 0.998608 |
| Reload parity: XGBoost | PASS | Reloaded models match the saved holdout fixture |
| Reload parity: MLP | PASS | Reloaded models match the saved holdout fixture |
| Reload parity: Blend | PASS | Reloaded models match the saved holdout fixture |
| Executed notebook: 01_eda.ipynb | PASS | 6 executed cells with no recorded errors |
| Executed notebook: 02_preprocessing.ipynb | PASS | 5 executed cells with no recorded errors |
| Executed notebook: 03_model_application.ipynb | PASS | 6 executed cells with no recorded errors |
| PDF: Water_Quality_Project_Report.pdf | PASS | 10 readable pages |
| PDF: Water_Quality_Literature_Review.pdf | PASS | 4 readable pages |
| PDF: Water_Quality_Prediction.pdf | PASS | 12 readable pages |
| Editable presentation | PASS | 12 slides, four native charts with embedded workbooks, 12 speaker-note parts |
| PPTX package integrity | PASS | Every declared content-type part exists |
| Source hash: Combined_dataset.csv | PASS | Original source file is unchanged |
| Source hash: Data_Processing_Harmonization.ipynb | PASS | Original source file is unchanged |
| Independent full formula replay | PASS | 2,827,977 rows; maximum absolute difference 2.842e-14 |
| Recorded training reproduction | PASS | Earlier independent full rerun: maximum metric difference 0.000e+00 |

Implementation tests and lint checks also run in GitHub Actions. Run python -m pytest -q and ruff check src tests scripts app.py locally.
Visual review covers the exported slide PDF and both report PDFs. Package checks alone do not prove appearance.
The scientific limitations remain part of the submitted report, rather than being treated as validation failures.
