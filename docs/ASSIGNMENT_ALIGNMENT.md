# Final check against Project 18

**Assessment:** appropriate for submission as a transparent index-emulation study
and AML/DL comparison. The brief explicitly permits a derived index if its target
dependence is acknowledged. Independent environmental prediction is not established.

| Assignment requirement | Implemented evidence | Assessment |
| --- | --- | --- |
| One documented class or index | Released continuous `CCME_Values`, exact formula in `docs/DATA.md` | Met, with explicit source-standard limitations |
| Avoid mixing incompatible standards | One source-defined row-level calculation; no alternate standard blended into labels | Met |
| Strong AML baseline | Regularized XGBoost, two depths, chronological validation, mean and Ridge references | Met |
| A small regularized MLP | PyTorch 64/32 and 128/64 candidates, ReLU, dropout, AdamW, early stopping | Met |
| Investigate a gap rather than assume DL wins | Same training sample, measured model comparison, XGBoost wins individually | Met |
| Fair site/time separation where possible | Train through 2018, validate 2019, calibrate 2020, test 2021–2023 | Met for time; no claim of unseen-site generalization |
| Regression metrics | Full-test MAE, RMSE, R-squared and subgroup results | Met |
| Missing-data treatment | Upstream imputation documented, new train-only imputation and fixed quality flags | Met within supplied-data constraints |
| Units and impossible values | Declared units and conservative flags, no guessed Fahrenheit conversion | Met, unresolved source units disclosed |
| Target leakage acknowledgement | Exact reconstruction of every source label, denominator 12 and upper-only tests disclosed | Met; central scientific limitation |
| Feature importance | Validation permutation importance and exact native TreeSHAP | Met; attribution is noncausal |
| Expected comparative report | Measured Markdown/HTML report and formal 10-page PDF | Met |
| Presentation and submission packaging | 12-slide editable deck, PDF, literature review and executed staged notebooks | Added for submission |
| Runnable demonstration | Saved-model CLI and Streamlit form/CSV demo | Added and tested |
| Reproducibility | Data hashes, archives, exact environment, repeated training, fixture parity and CI | Verified |

## Conclusions supported by the evidence

XGBoost has the lowest test MAE. A validation-selected 90/10 tree/neural blend
slightly lowers RMSE while increasing MAE. The neural network's complexity does not
yield a superior individual result on this threshold-derived target. Feature reliance
is strongest for orthophosphate and ammonia. Global residual intervals substantially
under-cover some index bands, even when aggregate coverage is near 90%.

## Claims to avoid

Do not call R-squared classification accuracy. Do not claim drinking-water safety,
official CCME certification, causal pollution drivers, new-country performance,
unseen-site generalization or a statistically significant blend improvement. The
dataset's row-level adaptation is not the official site-period CCME definition.

## Administrative details

Team names, enrollment numbers, section and institution intentionally remain dummy
text at the user's request. Edit `submission/team.json` and rebuild the PDF/deck
when final identities are supplied. No dummy observations or performance values
appear in the research results.
