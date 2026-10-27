# Training, validation, calibration and testing explained

This guide maps the four data partitions to the implemented notebooks. It is a reading companion for the presentation and a reference when changing the experiment.

## What each partition does

| Partition | Dates | Available records | Actual use |
| --- | --- | ---: | --- |
| Training | 1940-2018 | 2,478,761 | A shared seed-42 sample of 300,000 fits model parameters and preprocessing |
| Validation | 2019 | 87,177 | 50,000 tune candidate settings; the full year selects the final model and blend |
| Calibration | 2020 | 49,182 | All records set each frozen model's prediction-interval radius |
| Test | 2021-2023 | 202,886 | All records evaluate the frozen choices and calibrated intervals |

"Available" means eligible records in that period after preparation. "Used" describes how many enter a particular computation. Sampling stays inside a partition; later records are never sampled into historical training.

Training teaches the model. Validation helps choose its settings. Calibration estimates the size of its errors to construct intervals. Testing checks the final workflow on later records.

## Training and validation example

Suppose two XGBoost candidates use tree depths 4 and 6. Each learns from the same training rows. Predictions on validation rows tell us which has lower RMSE. Validation also determines early-stopping checkpoints.

The validation rows do not directly update neural weights through backpropagation, but they do influence model selection. This is why they cannot also serve as an untouched final test.

The final blend weight is selected from 21 choices on full-year validation predictions. It is 90% XGBoost and 10% MLP in the saved run. The test set does not choose this weight.

## Calibration with a worked example

Assume a model is already fitted and its settings are frozen. On separate calibration records:

| Record | Model predicts | Released target | Absolute error |
| --- | ---: | ---: | ---: |
| A | 80 | 81 | 1 |
| B | 70 | 68 | 2 |
| C | 90 | 93 | 3 |

The errors show how far predictions were from the released scores. Calibration uses many such errors to choose a radius. It does not retrain the predictor or change its point estimates.

For a larger illustrative calibration set, suppose the selected radius is 3. A new prediction of 80 receives the interval 77 to 83. A prediction of 99 receives 96 to 100 because bounds are clipped to the score's 0-100 range.

The three rows above illustrate residuals only. Three observations are too few for a finite 90% radius under this implementation's finite-sample rule.

### The exact rule used in the notebook

For n calibration errors and alpha=0.1:

1. Compute the absolute residual of each prediction.
2. Compute rank = ceil((n + 1) * 0.9), using one-based ranks.
3. Take the error at that position in sorted order.
4. If the rank exceeds n, return an infinite radius.
5. Add and subtract the radius from predictions, then clip bounds to 0-100.

For 100 calibration records, rank is 91, so use the 91st smallest error. This is a corrected order statistic, rather than an arbitrary choice of maximum error or standard deviation.

In the saved experiment, the blend radius is approximately 0.2534 index points. A hypothetical blend score of 80 would therefore receive bounds near 79.7466 and 80.2534.

## Coverage and width answer different questions

Coverage is the fraction of actual target values inside the interval. Width measures how precise or broad that interval is.

The blend's nominal coverage is 90%, and observed test coverage is about 89.3%. However, coverage is only 45.2% for Marginal and 51.9% for Poor records. Overall coverage can conceal weak groups.

The average blend width is approximately 0.4002, less than twice the radius, because clipping shortens intervals close to 100.

These intervals describe errors in emulating the released index. They do not quantify all laboratory or environmental uncertainty. Formal conformal guarantees require assumptions such as exchangeability; temporal dependence and shift make empirical coverage essential.

## Why use chronological partitions?

Later-year observations can differ from earlier ones. Separating periods avoids fitting our preprocessing on future records and gives a later-period evaluation.

This does not make the task forecasting: each prediction still uses the same record's measurements to estimate that record's derived index. Forecasting would require earlier inputs paired with a later target at a defined horizon.

It also does not create a new-site evaluation. About 85.9% of test rows have a Country+Area identifier present in the full historical training partition.

## Code map

- [Preprocessing notebook](../Notebooks/Preprocessing.ipynb): configuration, `prepare`, `temporal_split`, `sample`, and training-fitted transforms.
- [Model notebook](../Notebooks/02_Model_Application.ipynb): `fit_mlp`, candidate selection, blend selection, `residual_quantile`, `intervals` and `interval_metrics`.
- [README results](../README.md#results): recorded metrics and limitations.
- [Prediction checklist](Prediction_Checklist.md): running inference and interpreting its outputs.

## Short answers for presentation

**Does calibration train the model again?** No. It uses held-out errors to set interval radii after fitting and selection.

**Can calibration guarantee every answer lies in its interval?** No. Nominal overall coverage is not a guarantee for every record or subgroup, and time shift affects assumptions.

**Does the time split remove every kind of leakage?** No. Our own transformations use training only, but the released target is calculated from its predictors and upstream source imputation cannot be undone.

**Why not use test errors to choose the radius?** That would use the final test outcomes to design the method being evaluated.

Documentation prepared with AI assistance and checked against the notebooks and recorded outputs. These explanations introduce no new model experiments.

