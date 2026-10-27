# Viva questions and concise answers

**What is the task?** Regression of the released `CCME_Values` index using eight
chemistry measurements. The scope is source-index emulation.

**Why XGBoost?** Boosted trees learn thresholds and interactions efficiently on
tables. The target formula itself contains upper-threshold tests.

**Why an MLP?** It tests learned nonlinear combinations with a different model
structure. ReLU layers, dropout, weight decay and validation stopping control capacity.

**Why no LSTM?** A row of chemistry variables is not a sequence. A forecasting model
would need timestamped histories, a forecast horizon and future targets.

**Why regression rather than classification?** A continuous, documented index is
available. We choose one target and use MAE, RMSE and R-squared consistently.

**What does R-squared 0.9986 mean?** Very small squared error relative to variation in
the test index. It is not classification accuracy or a probability that water is safe.

**How did we control leakage?** We fit transformations on the same 300,000 historical
training rows for every model and use later, separate periods for validation,
calibration and testing. The target still structurally depends on the input readings.

**What is the main data issue?** The released formula reproduces every target using
the same measurements. It divides by 12 and checks all objectives as upper limits.
It differs from the official CCME site-period method and cannot certify water safety.

**Why are there no missing cells?** The data were already harmonized and imputed
upstream. We cannot reconstruct original missingness or detection limits.

**Which model wins?** XGBoost has the lowest MAE. Validation RMSE selects a 90/10
tree/neural blend, with a small test RMSE improvement. This is not a significance result.

**What does SHAP tell us?** How inputs contribute to a tree prediction. It does not
identify causal drivers of pollution, especially when the target is a known formula.

**Are the intervals reliable?** We report empirical coverage. The blend covers 89.3%
overall for nominal 90% intervals, but only 45.2% of Marginal rows. One global radius
does not provide uniform reliability under temporal or subgroup shift.

**What would improve the research?** Independent outcomes, original observation
flags, documented local objectives, multiple seeds, rolling evaluation and a
prespecified reduced sensor panel or genuine forecasting target.
