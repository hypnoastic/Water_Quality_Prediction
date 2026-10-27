"""End-to-end experiment; only validation data selects models and blend weights."""

import json
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import DMatrix, XGBRegressor

from .data import (
    FEATURES,
    SHORT_NAMES,
    TARGET,
    prepare,
    quality_features,
    read_dataset,
    sample,
    sha256,
    source_formula,
    temporal_split,
)
from .evaluation import interval_metrics, intervals, metrics, residual_quantile
from .models import fit_mlp, predict_mlp, seed_everything


def dump_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def table_audit(raw):
    reconstructed = source_formula(raw)
    error = np.abs(reconstructed - raw[TARGET].to_numpy())
    q = quality_features(raw)
    numeric = raw[FEATURES].apply(pd.to_numeric, errors="coerce")
    return {
        "raw_rows": len(raw),
        "raw_columns": len(raw.columns),
        "country_counts": raw.Country.value_counts().to_dict(),
        "class_counts": raw.CCME_WQI.value_counts().to_dict(),
        "area_counts_by_country": raw.groupby("Country").Area.nunique().to_dict(),
        "missing_as_delivered": raw.isna().sum().to_dict(),
        "invalid_or_missing_by_feature": q[FEATURES].isna().sum().to_dict(),
        "rows_with_quality_flags": int(q[FEATURES].isna().any(axis=1).sum()),
        "feature_summary": json.loads(numeric.describe().to_json()),
        "most_frequent_reading": {
            col: {
                "value": float(numeric[col].mode().iloc[0]),
                "fraction": float(numeric[col].value_counts(normalize=True).iloc[0]),
            }
            for col in FEATURES
        },
        "formula_audit": {
            "denominator": 12,
            "mae": float(np.nanmean(error)),
            "max_absolute_error": float(np.nanmax(error)),
            "fraction_matching_at_1e_7": float(np.mean(error < 1e-7)),
            "finite_reconstructions": int(np.isfinite(reconstructed).sum()),
        },
        "class_threshold_disagreements": int(
            np.sum(
                pd.cut(
                    raw[TARGET],
                    [-0.001, 45, 65, 80, 95, 100.001],
                    right=False,
                    labels=["Poor", "Marginal", "Fair", "Good", "Excellent"],
                )
                .astype(str)
                .to_numpy()
                != raw.CCME_WQI.to_numpy()
            )
        ),
    }


def split_summary(part):
    return {
        "rows": len(part),
        "start": str(part.parsed_date.min().date()),
        "end": str(part.parsed_date.max().date()),
        "countries": part.Country.value_counts().to_dict(),
        "sites": int(part.groupby(["Country", "Area"]).ngroups),
        "mean_target": float(part[TARGET].mean()),
    }


def explain(models, x, y, out, seed):
    """Permutation importance on validation, plus exact native TreeSHAP for XGBoost."""
    rng = np.random.default_rng(seed)
    rows = []
    for name in ["XGBoost", "MLP"]:
        predict = models[name]
        baseline = metrics(y, predict(x))["mae"]
        for col, short in zip(FEATURES, SHORT_NAMES):
            deltas = []
            for _ in range(3):
                shuffled = x.copy()
                # Move a reading and its quality flag together.
                ids = rng.permutation(len(x))
                shuffled.loc[:, [col, f"{col}__missing"]] = x.iloc[ids][
                    [col, f"{col}__missing"]
                ].to_numpy()
                deltas.append(metrics(y, predict(shuffled))["mae"] - baseline)
            rows.append(
                {
                    "model": name,
                    "feature": short,
                    "mae_increase": float(np.mean(deltas)),
                    "repeat_std": float(np.std(deltas)),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(out / "permutation_importance.csv", index=False)
    return table


def run(config):
    from .reporting import build_report

    start = time.perf_counter()
    out, artifacts = Path(config["output"]), Path(config["artifacts"])
    out.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    seed_everything(config["seed"], config["threads"])
    print("Reading and auditing the complete combined dataset...", flush=True)
    raw = read_dataset(config["data"])
    audit = table_audit(raw)
    clean, removed = prepare(raw)
    audit.update(removed)
    splits = temporal_split(clean, config)
    manifest = {key: split_summary(part) for key, part in splits.items()}
    train = sample(splits["train"], config["max_train_rows"], config["seed"])
    tuning = sample(splits["validation"], config["max_tuning_rows"], config["seed"])
    manifest["training_rows_used"] = len(train)
    manifest["tuning_rows_used"] = len(tuning)
    train_sites = pd.MultiIndex.from_frame(splits["train"][["Country", "Area"]])
    test_sites = pd.MultiIndex.from_frame(splits["test"][["Country", "Area"]])
    manifest["test_rows_at_sites_seen_before_2019"] = float(test_sites.isin(train_sites).mean())
    x = quality_features(train)
    y = train[TARGET].to_numpy()
    xv, yv = quality_features(tuning), tuning[TARGET].to_numpy()
    # All transforms learn exclusively from the shared training sample.
    tree_pre = SimpleImputer(strategy="median", keep_empty_features=True)
    xt, xvt = tree_pre.fit_transform(x), tree_pre.transform(xv)
    # Concentrations span orders of magnitude. asinh is smooth and also handles
    # legitimate subzero temperatures without a learned clipping threshold.
    neural_pre = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        FunctionTransformer(np.arcsinh, validate=False),
        StandardScaler(),
    )
    xn, xvn = (
        neural_pre.fit_transform(x).astype("float32"),
        neural_pre.transform(xv).astype("float32"),
    )
    candidates, histories = [], {}
    best_xgb, best_mlp = None, None
    best_xgb_score, best_mlp_score = float("inf"), float("inf")
    for depth in config["xgb_depths"]:
        print(f"Training XGBoost, depth {depth}...", flush=True)
        tick = time.perf_counter()
        model = XGBRegressor(
            n_estimators=config["xgb_estimators"],
            max_depth=depth,
            learning_rate=0.05,
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=1.0,
            reg_lambda=5,
            tree_method="hist",
            max_bin=256,
            objective="reg:squarederror",
            eval_metric="rmse",
            early_stopping_rounds=40,
            random_state=config["seed"],
            n_jobs=config["threads"],
        )
        model.fit(xt, y, eval_set=[(xvt, yv)], verbose=100)
        score = metrics(yv, np.clip(model.predict(xvt), 0, 100))
        candidates.append(
            {
                "family": "XGBoost",
                "architecture": str(depth),
                **score,
                "seconds": time.perf_counter() - tick,
                "best_step": int(model.best_iteration + 1),
            }
        )
        histories[f"XGBoost_depth_{depth}"] = model.evals_result()["validation_0"]["rmse"]
        if score["rmse"] < best_xgb_score:
            best_xgb, best_xgb_score = model, score["rmse"]
    for widths in config["mlp_widths"]:
        print(f"Training MLP {widths}...", flush=True)
        tick = time.perf_counter()
        model, history = fit_mlp(xn, y, xvn, yv, widths, config)
        score = metrics(yv, predict_mlp(model, xvn))
        candidates.append(
            {
                "family": "MLP",
                "architecture": str(widths),
                **score,
                "seconds": time.perf_counter() - tick,
                "best_step": min(history, key=lambda r: r["validation_rmse"])["epoch"],
            }
        )
        histories[f"MLP_{widths}"] = history
        if score["rmse"] < best_mlp_score:
            best_mlp, best_mlp_score, best_widths = model, score["rmse"], widths
    pd.DataFrame(candidates).to_csv(out / "validation_candidates.csv", index=False)
    dump_json(out / "training_history.json", histories)
    dummy = DummyRegressor(strategy="mean").fit(xt, y)
    ridge = Ridge(alpha=10).fit(xn, y)

    def tree_predict(frame):
        return np.clip(best_xgb.predict(tree_pre.transform(frame)), 0, 100)

    def neural_predict(frame):
        return predict_mlp(best_mlp, neural_pre.transform(frame).astype("float32"))

    models = {
        "Mean baseline": lambda frame: dummy.predict(tree_pre.transform(frame)),
        "Ridge": lambda frame: np.clip(ridge.predict(neural_pre.transform(frame)), 0, 100),
        "XGBoost": tree_predict,
        "MLP": neural_predict,
    }
    # The blend weight is selected on validation only; no assumption that mixing helps.
    val_full_x = quality_features(splits["validation"])
    val_full_y = splits["validation"][TARGET].to_numpy()
    px, pn = tree_predict(val_full_x), neural_predict(val_full_x)
    weights = np.linspace(0, 1, 21)
    weight = float(min(weights, key=lambda w: metrics(val_full_y, w * px + (1 - w) * pn)["rmse"]))
    models["Blend"] = lambda frame: (
        weight * tree_predict(frame) + (1 - weight) * neural_predict(frame)
    )
    validation_scores = {
        name: metrics(val_full_y, predict(val_full_x)) for name, predict in models.items()
    }
    # Keep the simpler named family if a blend degenerates to an endpoint.
    eligible = ["XGBoost", "MLP"] + (["Blend"] if 0 < weight < 1 else [])
    champion = min(eligible, key=lambda name: validation_scores[name]["rmse"])
    print(f"Validation-selected model: {champion}; blend XGBoost weight = {weight:.2f}", flush=True)
    cal_x, cal_y = quality_features(splits["calibration"]), splits["calibration"][TARGET].to_numpy()
    test = splits["test"]
    test_x, test_y = quality_features(test), test[TARGET].to_numpy()
    score_rows, subgroups, radii, predictions = [], [], {}, {}
    prediction_table = test[["source_row", "Country", "Area", "Date", TARGET]].copy()
    for name, predict in models.items():
        print(f"Evaluating frozen {name} on the full chronological holdout...", flush=True)
        prediction = predict(test_x)
        predictions[name] = prediction
        radius = residual_quantile(cal_y, predict(cal_x), config["alpha"])
        radii[name] = radius
        score_rows.append(
            {
                "model": name,
                **metrics(test_y, prediction),
                **interval_metrics(test_y, prediction, radius),
                "validation_rmse": validation_scores[name]["rmse"],
            }
        )
        prediction_table[name] = prediction
        for country in sorted(test.Country.unique()):
            ids = (test.Country == country).to_numpy()
            subgroups.append(
                {
                    "model": name,
                    "group_type": "country",
                    "group": country,
                    "rows": int(ids.sum()),
                    **metrics(test_y[ids], prediction[ids]),
                    **interval_metrics(test_y[ids], prediction[ids], radius),
                }
            )
        for band in ["Poor", "Marginal", "Fair", "Good", "Excellent"]:
            ids = (test.CCME_WQI == band).to_numpy()
            if ids.sum() >= 2:
                subgroups.append(
                    {
                        "model": name,
                        "group_type": "source_index_band",
                        "group": band,
                        "rows": int(ids.sum()),
                        **metrics(test_y[ids], prediction[ids]),
                        **interval_metrics(test_y[ids], prediction[ids], radius),
                    }
                )
    lower, upper = intervals(predictions[champion], radii[champion])
    prediction_table["selected_lower_90"] = lower
    prediction_table["selected_upper_90"] = upper
    prediction_table.to_csv(out / "predictions_test.csv.gz", index=False, compression="gzip")
    scores, groups = pd.DataFrame(score_rows), pd.DataFrame(subgroups)
    scores.to_csv(out / "metrics.csv", index=False)
    groups.to_csv(out / "subgroup_metrics.csv", index=False)
    print("Computing validation permutation importance and native TreeSHAP...", flush=True)
    explain_frame = sample(splits["validation"], config["explanation_rows"], config["seed"])
    explain_x = quality_features(explain_frame)
    importance = explain(models, explain_x, explain_frame[TARGET].to_numpy(), out, config["seed"])
    # Native TreeSHAP returns contributions in index units, plus the bias column.
    booster = best_xgb.get_booster()
    contributions = booster.predict(
        DMatrix(tree_pre.transform(explain_x)),
        pred_contribs=True,
        iteration_range=(0, best_xgb.best_iteration + 1),
    )
    shap_table = pd.DataFrame(
        {
            "feature": SHORT_NAMES,
            "mean_absolute_shap": np.mean(np.abs(contributions[:, :8]), axis=0),
            "missing_flag_mean_absolute_shap": np.mean(np.abs(contributions[:, 8:16]), axis=0),
        }
    )
    shap_table.to_csv(out / "shap_importance.csv", index=False)
    # This is a formula diagnostic on unmodified source inputs, never an ML candidate.
    audit["formula_test_metrics"] = metrics(test_y, source_formula(test))
    audit["retained_rows"] = len(clean)
    dump_json(out / "data_audit.json", audit)
    dump_json(out / "split_manifest.json", manifest)
    best_xgb.save_model(artifacts / "xgboost.json")
    torch.save(best_mlp.state_dict(), artifacts / "mlp.pt")
    joblib.dump({"tree": tree_pre, "neural": neural_pre}, artifacts / "preprocessing.joblib")
    metadata = {
        "schema_version": 1,
        "selected_model": champion,
        "features": FEATURES,
        "mlp_widths": best_widths,
        "dropout": 0.1,
        "blend_xgboost_weight": weight,
        "radii": radii,
        "alpha": config["alpha"],
        "target": TARGET,
        "task": "Emulation of the source-defined, row-level derived index",
        "xgboost_best_iteration": int(best_xgb.best_iteration),
        "warning": "This index is not official site-period CCME WQI or drinking-water safety.",
    }
    dump_json(artifacts / "metadata.json", metadata)
    dump_json(
        artifacts / "checksums.json",
        {
            name: sha256(artifacts / name)
            for name in ["xgboost.json", "mlp.pt", "preprocessing.joblib", "metadata.json"]
        },
    )
    # Small, independent inference fixture; original field names and units are retained.
    example = test.sample(n=min(5, len(test)), random_state=config["seed"])
    Path("examples").mkdir(exist_ok=True)
    example[FEATURES].to_csv("examples/measurements.csv", index=False)
    prediction_table.set_index("source_row").loc[example.source_row].reset_index().to_csv(
        "examples/expected_predictions.csv", index=False
    )
    provenance = {
        "created_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in [
                "numpy",
                "pandas",
                "scikit-learn",
                "xgboost",
                "torch",
                "matplotlib",
                "joblib",
            ]
        },
        "config": config,
        "dataset_sha256": sha256(config["data"]),
        "reference_notebook_sha256": {
            p.name: sha256(p) for p in Path("notebooks/reference").glob("*.ipynb")
        },
        "runtime_seconds_before_report": time.perf_counter() - start,
        "validation_scores": validation_scores,
    }
    dump_json(out / "run_manifest.json", provenance)
    build_report(
        out,
        audit,
        manifest,
        scores,
        groups,
        importance,
        shap_table,
        histories,
        prediction_table,
        metadata,
        provenance,
    )
    print(scores[["model", "mae", "rmse", "r2", "coverage"]].to_string(index=False), flush=True)
    print(f"Finished. Read {out / 'report.md'}", flush=True)
