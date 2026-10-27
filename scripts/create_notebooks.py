"""Create the staged academic notebooks and optionally execute them in process.

Run from the repository root. --execute records real cell outputs without starting
a network-bound Jupyter kernel, which also works in restricted local environments.
"""

import argparse
import os
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SETUP = """from pathlib import Path
import json
import pandas as pd
import numpy as np
from IPython.display import display, Image
ROOT = Path.cwd() if (Path.cwd() / 'configs/default.json').exists() else Path.cwd().parent
REPORTS = ROOT / 'reports'
config = json.loads((ROOT / 'configs/default.json').read_text())
"""


def notebook(title, sections):
    cells = [nbformat.v4.new_markdown_cell(title), nbformat.v4.new_code_cell(SETUP)]
    for explanation, source in sections:
        cells.extend(
            [nbformat.v4.new_markdown_cell(explanation), nbformat.v4.new_code_cell(source)]
        )
    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (water quality)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )


def execute(book):
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output

    shell = InteractiveShell.instance(user_ns={})
    shell.reset(new_session=False)
    count = 0
    for cell in book.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(cell.source, store_history=False)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(str(result.error_before_exec or result.error_in_exec))
        cell.execution_count = count
        cell.outputs = []
        if captured.stdout:
            cell.outputs.append(
                nbformat.v4.new_output("stream", name="stdout", text=captured.stdout)
            )
        if captured.stderr:
            cell.outputs.append(
                nbformat.v4.new_output("stream", name="stderr", text=captured.stderr)
            )
        cell.outputs.extend(
            nbformat.v4.new_output("display_data", data=o.data, metadata=o.metadata)
            for o in captured.outputs
        )
    return book


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault("IPYTHONDIR", str(ROOT / ".cache/ipython"))
    books = {
        "02_eda.ipynb": notebook(
            "# Exploratory data analysis\n\nProject 18: inspect the supplied water-quality data before fitting a model. This notebook requires the combined raw CSV. Restore it with `python scripts/download_data.py`. The target is the authors' calculated index, so interpretation depends on its provenance.",
            [
                (
                    "## 1. Load and inspect the actual source file\n\nThe combined CSV is the single source. Country CSVs are alternate views, not extra observations.",
                    "from water_quality.data import read_dataset, FEATURES, TARGET\ndf = read_dataset(ROOT / config['data'])\nprint('Shape:', df.shape)\ndisplay(df.head())\ndisplay(df.Country.value_counts().rename('Rows'))",
                ),
                (
                    "## 2. Missingness, ranges and duplicates\n\nNo empty cell does not mean no upstream imputation. Preserve extreme pollution events and inspect units rather than deleting every IQR outlier.",
                    "display(df.isna().sum().rename('Empty cells'))\ndisplay(df[FEATURES + [TARGET]].describe().T)\nprint('Exact duplicates:', df.duplicated().sum())\nprint('Most frequent readings:')\nfor column in FEATURES:\n    mode = df[column].value_counts().head(1)\n    print(column, mode.to_dict())",
                ),
                (
                    "## 3. Representation and metadata\n\nEngland dominates the sample. Area is a coarse proxy for a site. A temporal split does not prove unseen-site generalization.",
                    "dates = pd.to_datetime(df.Date, format='%d-%m-%Y', errors='coerce')\nprint('Date range:', dates.min(), dates.max())\nprint('Invalid dates:', dates.isna().sum())\ndisplay(df.groupby('Country').Area.nunique().rename('Distinct Area labels'))\ndisplay(Image(filename=str(REPORTS / 'figures/data_overview.png')))",
                ),
                (
                    "## 4. Audit the target formula\n\nThe reconstruction is a provenance check. It is deliberately separate from trained-model performance. The exact denominator is 12 and all tests use upper objectives.",
                    "from water_quality.data import source_formula, quality_features\nerror = np.abs(source_formula(df) - df[TARGET].to_numpy())\nprint('Matching within 1e-7:', (error < 1e-7).mean())\nprint('Maximum difference:', error.max())\ndisplay(quality_features(df)[FEATURES].isna().sum().rename('Missing or flagged'))",
                ),
                (
                    "## 5. Decisions\n\nUse one continuous target. Exclude the redundant class label and metadata from features. Remove exact duplicates, keep pollution extremes and use chronological partitions. Document the derived-target limitation.",
                    "audit = json.loads((REPORTS / 'data_audit.json').read_text())\nassert len(df) == audit['raw_rows']\nassert (error < 1e-7).all()\nprint('EDA and stored audit agree.')",
                ),
            ],
        ),
        "03_preprocessing.ipynb": notebook(
            "# Preprocessing and experiment design\n\nProject 18: reproduce the actual partitions and training-only transforms. Run EDA first to understand the target and source limitations.",
            [
                (
                    "## 1. Clean metadata and create chronological partitions\n\nNever impute the target or use random row splitting for this benchmark.",
                    "from water_quality.data import read_dataset, prepare, temporal_split, sample, quality_features, FEATURES\nraw = read_dataset(ROOT / config['data'])\nclean, removed = prepare(raw)\nparts = temporal_split(clean, config)\nprint('Removed:', removed)\ndisplay(pd.DataFrame({name: {'rows':len(part), 'start':str(part.parsed_date.min().date()), 'end':str(part.parsed_date.max().date())} for name,part in parts.items()}).T)",
                ),
                (
                    "## 2. Share the exact training sample across models\n\nThe same seeded 300,000 rows train all model families. Candidate tuning uses a seeded 50,000-row validation subset. The complete validation period chooses the family and blend.",
                    "train = sample(parts['train'], config['max_train_rows'], config['seed'])\ntune = sample(parts['validation'], config['max_tuning_rows'], config['seed'])\nx = quality_features(train)\nprint('Training rows:', len(train), 'Tuning rows:', len(tune))\nprint('Model input dimensions:', x.shape[1])\ndisplay(x.head())",
                ),
                (
                    "## 3. Verify training-fitted preprocessing\n\nInvalid inputs become missing and receive fixed flags. Tree models use median-imputed values. MLP/Ridge add asinh and standard scaling. Loading joblib is safe only for trusted local artifacts.",
                    "import joblib\npre = joblib.load(ROOT / 'artifacts/preprocessing.joblib')\nmanual_medians = np.nanmedian(x.to_numpy(), axis=0)\nmanual_medians = np.nan_to_num(manual_medians, nan=0)\nnp.testing.assert_allclose(pre['tree'].statistics_, manual_medians)\nxt = pre['tree'].transform(x.iloc[:5])\nxn = pre['neural'].transform(x.iloc[:5])\nprint('Tree transform shape:', xt.shape, 'Neural shape:', xn.shape)\nprint('Stored medians match the historical training sample.')",
                ),
                (
                    "## 4. Verify disjoint rows and chronological boundaries\n\nHistorical stations can recur. Strictly separated dates prevent future rows from entering historical fitting, but cannot undo preprocessing already performed by the dataset authors.",
                    "for name,part in parts.items():\n    print(name, len(part), part.parsed_date.min(), part.parsed_date.max())\nsets = [set(part.source_row) for part in parts.values()]\nassert sum(map(len,sets)) == len(set.union(*sets))\nordered = list(parts.values())\nassert all(left.parsed_date.max() < right.parsed_date.min() for left,right in zip(ordered, ordered[1:]))\nprint('All partitions have disjoint source rows and ordered dates.')",
                ),
            ],
        ),
        "04_model_application.ipynb": notebook(
            "# Model application and final comparison\n\nProject 18: examine trained candidates, evaluate the recorded chronological holdout and run saved-model inference. This notebook works without the raw CSV. Retraining uses the tested CLI so the modeling code has one implementation.",
            [
                (
                    "## 1. Model selection uses validation\n\nXGBoost tests depths 4/6. MLP tests 64/32 and 128/64 hidden widths with dropout, weight decay and early stopping. The final family and blend use full-validation RMSE.",
                    "display(pd.read_csv(REPORTS / 'validation_candidates.csv'))\nmetadata = json.loads((ROOT / 'artifacts/metadata.json').read_text())\nprint('Selected:', metadata['selected_model'])\nprint('XGBoost blend weight:', metadata['blend_xgboost_weight'])\nprint('Selected MLP widths:', metadata['mlp_widths'])\ndisplay(Image(filename=str(REPORTS / 'figures/learning_curves.png')))",
                ),
                (
                    "## 2. Complete future test period\n\nMAE/RMSE use index points. R-squared is not classification accuracy. XGBoost has the lowest MAE. The blend changes RMSE slightly, without a significance claim.",
                    "metrics = pd.read_csv(REPORTS / 'metrics.csv')\ndisplay(metrics)\ndisplay(Image(filename=str(REPORTS / 'figures/actual_vs_predicted.png')))",
                ),
                (
                    "## 3. Explain reliance and evaluate uncertainty\n\nPermutation importance and native TreeSHAP explain model predictions. The 2020-calibrated global intervals under-cover several target bands in the later period.",
                    "display(pd.read_csv(REPORTS / 'permutation_importance.csv'))\ngroups = pd.read_csv(REPORTS / 'subgroup_metrics.csv')\ndisplay(groups[(groups.model == 'Blend') & (groups.group_type == 'source_index_band')][['group','rows','mae','coverage']])\ndisplay(Image(filename=str(REPORTS / 'figures/country_performance.png')))",
                ),
                (
                    "## 4. Reload every learned model\n\nThese are real predictions from saved models and the exact training-fitted transforms. Compare them with the independent fixture retained from the test run.",
                    "from tempfile import TemporaryDirectory\nfrom water_quality.predict import predict_file\nexpected = pd.read_csv(ROOT / 'examples/expected_predictions.csv')\nfor name in ['XGBoost','MLP','Blend']:\n    with TemporaryDirectory() as directory:\n        result = predict_file(ROOT / 'examples/measurements.csv', ROOT / 'artifacts', Path(directory) / 'predictions.csv', name)\n        np.testing.assert_allclose(result.predicted_source_index, expected[name], atol=1e-4, rtol=1e-6)\n        display(result[['predicted_source_index','lower_90','upper_90','model']])\nprint('All three models reproduce the recorded test fixture.')",
                ),
                (
                    "## 5. Reproduce training\n\nRun `python -m water_quality train --config configs/default.json` from the repository root. It audits the source, trains both candidate families, evaluates the frozen selection and rebuilds the reports. Preserve the delivered outputs by changing the output/artifact paths for experiments.",
                    "print('Training command: python -m water_quality train --config configs/default.json')\nprint('Demo command: streamlit run app.py')\nprint('Verification command: python scripts/verify_submission.py')",
                ),
            ],
        ),
    }
    for name, book in books.items():
        if args.execute:
            book = execute(book)
        nbformat.validate(book)
        nbformat.write(book, ROOT / "notebooks" / name)
        print(f"{'Executed and saved' if args.execute else 'Created'} {name}")


if __name__ == "__main__":
    main()
