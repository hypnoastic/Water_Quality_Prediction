# Saved-model prediction checklist

Use this alongside [Model Application](../Notebooks/02_Model_Application.ipynb). For the meaning of calibration, see [Workflow Explained](Workflow_Explained.md).

## Run without retraining

1. Open the model notebook in the environment described in the [README](../README.md#run-the-project).
2. Run the imports, model class and helper definitions in section 1.
3. In section 2, set `RETRAIN = False` before executing its code. This extracts the included `water_quality_project_outputs.zip` into `Datasets/_working/`.
4. Run the remaining notebook cells in order. Training blocks guarded by `if RETRAIN` are skipped; saved results can still be displayed.
5. In section 10, the code checks artifact hashes, reloads preprocessors and weights, and defines `predict_measurements`.
6. First run the provided five-row example. Its assertions compare reloaded predictions with recorded holdout examples.
7. Then call `predict_measurements` on your own dataframe with the headers below. Do not refit preprocessing on these new rows.

This procedure describes the existing workflow. It is not a new claim that the models were retrained or independently validated during documentation work.

## Input contract

| Exact CSV column name | Source unit |
| --- | --- |
| `Ammonia (mg/l)` | mg/L |
| `Biochemical Oxygen Demand (mg/l)` | mg/L |
| `Dissolved Oxygen (mg/l)` | mg/L |
| `Orthophosphate (mg/l)` | mg/L |
| `pH (ph units)` | Dimensionless |
| `Temperature (cel)` | Celsius |
| `Nitrogen (mg/l)` | mg/L |
| `Nitrate (mg/l)` | mg/L |

All eight columns must exist. Country, Area, Date and target columns are not required for inference. The code selects its own feature order by column name. A missing required column raises an error.

After section 10 defines the prediction function:

```python
new_readings = pd.read_csv("your_measurements.csv")
predictions = predict_measurements(new_readings)
display(predictions)
```

## How a row becomes model inputs

1. `quality_features` converts the eight readings to numbers and masks questionable entries.
2. Negative concentrations, nonfinite values and pH outside 0-14 are masked. Temperature outside -5 to 50 Celsius is masked under the declared screening heuristic, not a universal physical law.
3. Eight flags record which readings were missing or invalid, producing 16 model inputs.
4. Saved training medians fill masked values. XGBoost receives these imputed inputs.
5. The MLP additionally receives the saved asinh transformation and StandardScaler. In this implementation they apply to all 16 inputs, including flags.
6. Saved models predict scores. The selected blend combines 90% XGBoost with 10% MLP.

Do not guess Fahrenheit conversions from an unusual temperature. Confirm units with the data source. Also, a large positive concentration is not automatically removed just because it is unusual: it may represent genuine pollution.

## ASINH versus scaling

ASINH is the inverse hyperbolic sine, implemented as `np.arcsinh`. It compresses large magnitudes while preserving their sign and ordering.

| Input | Approximate ASINH output |
| ---: | ---: |
| -100 | -5.2983 |
| -1 | -0.8814 |
| 0 | 0 |
| 1 | 0.8814 |
| 10 | 2.9982 |
| 100 | 5.2983 |
| 1000 | 7.6009 |

Large negatives are compressed too. ASINH does not force every value close to zero or into 0-1. Values near zero change relatively little.

StandardScaler subsequently subtracts each transformed input's training mean and divides by its training standard deviation. Min-max scaling is a different transformation and is not used here. ASINH changes relative spacing nonlinearly; standard scaling is an affine rescaling.

An illustration such as `asinh(-100)` describes the mathematical function. In this project's actual pipeline, invalid negative concentrations and temperatures outside the screening range are masked before that transformation.

## Output contract

| Output | Meaning |
| --- | --- |
| `XGBoost` | Clipped tree-model score |
| `MLP` | Neural score restored to index points and clipped to 0-100 |
| `Blend` | Weighted combination of the two model scores |
| `selected_lower_90` | Selected model's lower calibrated bound, clipped at 0 |
| `selected_upper_90` | Selected model's upper calibrated bound, clipped at 100 |
| `flagged_inputs` | Number of readings masked or missing after screening |

Bounds are for the validation-selected model recorded in `metadata.json`. A flag count is a data-quality indicator, not a pollution or safety class.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Repository-root error | Start inside the repository or its Notebooks folder so `Datasets/` can be found |
| `prepared.joblib` is missing | For retraining, run EDA then Preprocessing first; for saved inference, set `RETRAIN=False` before section 2 |
| `predict_measurements` is undefined | Run section 10 after imports and archive loading |
| Missing required measurement columns | Compare spelling, capitalization, spaces and units with the exact input contract |
| Some inputs are flagged | Inspect source readings and units; masking and imputation do not validate a new missing-sensor scenario |
| Artifact checksum mismatch | Investigate changed files and restore the intended trusted archive rather than bypass the check |

## Interpretation boundaries

The output emulates the released same-row formula. It is not a next-day forecast, an official site-period CCME assessment or a drinking-water safety decision. When complete original formula components are available, direct calculation is preferable for reproducing this exact index.

The model can technically impute missing readings, but reliable performance with an absent sensor has not been demonstrated. A reduced-sensor task needs separate training and evaluation. Overall interval coverage also does not establish coverage for each subgroup or new location.

Documentation prepared with AI assistance and checked against the notebook implementation and saved metadata. No new model experiments are claimed.
