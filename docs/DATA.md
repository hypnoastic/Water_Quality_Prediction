# Data card and target audit

## Provenance

Source: Karim et al., [Figshare 27800394, v2](https://doi.org/10.6084/m9.figshare.27800394.v2),
published 23 February 2025; **CC BY 4.0**. The supplied files are the harmonized
release, not original laboratory observations. The combined CSV contains
**2,827,977 rows and 14 columns**, dated 2 January 1940–17 December 2023.

| Country | Rows | Distinct `Area` labels |
| --- | ---: | ---: |
| England | 2,129,198 | 48,642 |
| USA | 413,814 | 955 |
| Ireland | 235,019 | 9,737 |
| China | 45,997 | 1 |
| Canada | 3,949 | 2,354 |

`Country`, `Area`, `Waterbody Type` and `Date` are metadata. `Area` is only a coarse
site proxy, particularly for China. Dates are parsed strictly as **day-month-year**.
Country-specific files are alternate views of the combined data; do not append them
to it. Original summary spreadsheets are descriptive references, not model inputs.

## Schema

| Input column | Declared unit | Source formula's upper objective |
| --- | --- | ---: |
| `Ammonia (mg/l)` | mg/L | 0.5 |
| `Biochemical Oxygen Demand (mg/l)` | mg/L | 6 |
| `Dissolved Oxygen (mg/l)` | mg/L | 14 |
| `Orthophosphate (mg/l)` | mg/L | 0.05 |
| `pH (ph units)` | unitless | 8.5 |
| `Temperature (cel)` | °C | 25 |
| `Nitrogen (mg/l)` | mg/L | 20 |
| `Nitrate (mg/l)` | mg/L | 45 |

These objectives reproduce the **authors' code**. They are not asserted to be
universally appropriate ecological or drinking-water limits. No external standard
is combined with them. The target is `CCME_Values`; `CCME_WQI` is a redundant band
label and is never supplied to a model.

## Exact reconstruction of the released target

For each row with eight measurements `x[j]` and the objectives `s[j]` above:

```text
failed = sum(x[j] > s[j])
excursion = sum(max(x[j] / s[j] - 1, 0))
F1 = F2 = 100 * failed / 12
nse = excursion / 12
F3 = nse / (0.01 * nse + 0.01)
index = 100 - sqrt(F1² + F2² + F3²) / 1.732
```

This matches **all source rows within 1e-7**, before our quality filtering.
`data.py::source_formula` preserves these choices deliberately as an audit reference.
The denominator **12** is established empirically from the released file. The source
notebook computes `columns = len(df.columns)`, so the denominator depends on which
metadata columns exist at execution. With eight parameter tests, 12 is not a
measurement-count denominator. We neither silently repair nor relabel the target.

The [official CCME manual](https://ccme.ca/en/res/wqimanualen.pdf) defines scope,
frequency and amplitude across specified variables, objectives and a sampling
period. Its objectives may be upper limits, lower limits or ranges. The supplied
code instead uses one row, equates frequency with scope, treats dissolved oxygen
as upper-only and omits a lower pH objective. Therefore use the phrase
**source-defined row-level index**, rather than claiming certified CCME WQI.

The bands in the delivered file are Poor `[0,45)`, Marginal `[45,65)`, Fair `[65,80)`,
Good `[80,95)` and Excellent `[95,100]`. They are used only to describe error slices.

## Quality handling and limitations

- There are **no empty cells** in the delivered combined data. The source preprocessing
  notebook shows median imputation before modeling. Original missingness cannot be
  recovered; frequent repeated values are consistent with substantial preprocessing.
- **9,971 exact duplicates** are removed before splitting, retaining the first original
  row ID. Different measurements at the same area/date are retained: identifiers do
  not reliably identify laboratory sampling events.
- Negative concentrations, nonfinite inputs and pH outside `[0,14]` become missing.
  Temperature outside `[-5,50] °C` is flagged for review. This is a declared screening
  range, not a universal physical bound. No undocumented unit conversion is guessed.
- Each feature receives a missing/invalid flag. Training medians impute inputs.
  Original source targets are retained; invalid targets or dates are excluded.
  High positive concentration extremes are preserved, rather than removed by IQR.
- Values such as USA temperature `63` under a Celsius header raise a unit question.
  The source notebook's Fahrenheit conversion is commented out; the supplied data do
  not establish which values need conversion.
- Detection-limit qualifiers and original sampling procedures are unavailable. New
  strings such as `<0.01` are flagged, never silently converted into exact measurements.
- Metadata are used for time splits and subgroup auditing only. Historical sites can
  recur in the test period. Upstream imputation may have used information across our
  split boundary; a downstream pipeline cannot undo this limitation.

Machine-readable counts, ranges, modal values and reconstruction errors are in
[`reports/data_audit.json`](../reports/data_audit.json). The exact combined-file hash
is in [`reports/run_manifest.json`](../reports/run_manifest.json).

## Original-code review

`Data_Processing_Harmonization.ipynb` demonstrates translation, conditional unit
conversion, median imputation and the derived label. It references external Colab
inputs not supplied here; it is preserved for provenance, not advertised as runnable.

`Technical_Validation.ipynb` uses hardcoded Kaggle paths, removes outliers before the
split, scales before a random split, and treats tabular columns as LSTM time steps.
The new pipeline replaces those evaluation choices with explicit partitions,
training-only transforms, retained pollution extremes and an appropriate tabular MLP.
