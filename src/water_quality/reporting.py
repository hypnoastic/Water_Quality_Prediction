"""Publication-style figures and a portable, offline HTML research report."""

import os
from html import escape
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "Mean baseline": "#9aaeb2",
    "Ridge": "#6893a1",
    "XGBoost": "#087f8c",
    "MLP": "#e09540",
    "Blend": "#6a63a6",
}


def markdown_table(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = [
            f"{row[c]:.4f}" if isinstance(row[c], (float, np.floating)) else str(row[c])
            for c in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def save(fig, directory, name):
    fig.savefig(directory / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_figures(out, audit, splits, scores, groups, importance, shap, histories, predictions):
    directory = out / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#314a52",
            "text.color": "#163640",
            "axes.titleweight": "bold",
            "axes.titlepad": 14,
            "grid.color": "#dfe9ea",
            "axes.axisbelow": True,
        }
    )
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    countries = pd.Series(audit["country_counts"]).sort_values()
    ax[0].barh(countries.index, countries.values, color="#087f8c")
    ax[0].set(
        xlabel="Rows (log scale)", title="Five countries, unequal representation", xscale="log"
    )
    for i, count in enumerate(countries.values):
        ax[0].text(count * 1.05, i, f"{count:,}", va="center", fontsize=9)
    ax[0].set_xlim(1000, countries.max() * 3)
    classes = pd.Series(audit["class_counts"]).reindex(
        ["Poor", "Marginal", "Fair", "Good", "Excellent"]
    )
    ax[1].bar(
        classes.index,
        classes.values / 1000,
        color=["#b55750", "#df9862", "#c6b775", "#71a99d", "#087f8c"],
    )
    ax[1].set(ylabel="Rows (thousands)", title="Distribution of the released index bands")
    save(fig, directory, "data_overview.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), layout="constrained")
    for ax, metric, label in zip(axes, ["mae", "rmse", "r2"], ["MAE ↓", "RMSE ↓", "R² ↑"]):
        ax.barh(scores.model, scores[metric], color=[COLORS[n] for n in scores.model])
        ax.set(xlabel=label, title=f"Chronological test · {label}")
        ax.grid(axis="x", alpha=0.6)
    save(fig, directory, "model_comparison.png")

    plot_data = predictions.sample(n=min(7000, len(predictions)), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, name in zip(axes, ["XGBoost", "MLP"]):
        ax.hexbin(
            plot_data.CCME_Values,
            plot_data[name],
            gridsize=45,
            bins="log",
            cmap="YlGnBu",
            mincnt=1,
            extent=(25, 101, 25, 101),
        )
        ax.plot([25, 101], [25, 101], color="#b96c47", linestyle="--", linewidth=1)
        ax.set(
            xlabel="Released source index",
            ylabel="Predicted source index",
            title=name,
            xlim=(25, 101),
            ylim=(25, 101),
        )
    save(fig, directory, "actual_vs_predicted.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, name in zip(axes, ["XGBoost", "MLP"]):
        part = importance[importance.model == name].sort_values("mae_increase")
        ax.barh(
            part.feature, part.mae_increase, xerr=part.repeat_std, color=COLORS[name], capsize=3
        )
        ax.set(
            xlabel="Increase in validation MAE after shuffling",
            title=f"{name} · permutation importance",
        )
        ax.grid(axis="x", alpha=0.5)
    save(fig, directory, "feature_importance.png")

    fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
    part = shap.sort_values("mean_absolute_shap")
    ax.barh(part.feature, part.mean_absolute_shap, color="#087f8c")
    ax.set(
        xlabel="Mean |TreeSHAP contribution| (index points)",
        title="How XGBoost reconstructs the source index",
    )
    save(fig, directory, "shap_importance.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for name, history in histories.items():
        if name.startswith("MLP"):
            axes[0].plot(
                [r["epoch"] for r in history], [r["validation_rmse"] for r in history], label=name
            )
        else:
            axes[1].plot(np.arange(len(history)) + 1, history, label=name)
    for ax, title, xlabel in zip(
        axes, ["MLP early stopping", "XGBoost early stopping"], ["Epoch", "Boosting round"]
    ):
        ax.set(xlabel=xlabel, ylabel="Validation RMSE", title=title, yscale="log")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.5)
    save(fig, directory, "learning_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    part = groups[(groups.group_type == "country") & groups.model.isin(["XGBoost", "MLP"])]
    countries = sorted(part.group.unique())
    pos = np.arange(len(countries))
    for offset, name in zip([-0.18, 0.18], ["XGBoost", "MLP"]):
        rows = part[part.model == name].set_index("group").reindex(countries)
        axes[0].bar(pos + offset, rows.mae, width=0.35, label=name, color=COLORS[name])
        axes[1].bar(pos + offset, rows.coverage * 100, width=0.35, label=name, color=COLORS[name])
    for ax in axes:
        ax.set_xticks(pos, countries)
        ax.legend()
        ax.grid(axis="y", alpha=0.4)
    axes[0].set(ylabel="MAE (index points)", title="Test error by represented country")
    axes[1].set(
        ylabel="Empirical coverage (%)", title="90% intervals under temporal shift", ylim=(0, 102)
    )
    axes[1].axhline(90, color="#a45d4f", linestyle="--", linewidth=1)
    save(fig, directory, "country_performance.png")


def build_report(
    out,
    audit,
    splits,
    scores,
    groups,
    importance,
    shap,
    histories,
    predictions,
    metadata,
    provenance,
):
    create_figures(out, audit, splits, scores, groups, importance, shap, histories, predictions)
    selected = metadata["selected_model"]
    row = scores.set_index("model").loc[selected]
    formula = audit["formula_audit"]
    split_frame = pd.DataFrame(
        [
            {"partition": key, **splits[key]}
            for key in ["train", "validation", "calibration", "test"]
        ]
    )
    country = groups[(groups.group_type == "country") & (groups.model == selected)]
    band = groups[(groups.group_type == "source_index_band") & (groups.model == selected)]
    findings = (
        f"{selected} was selected using validation RMSE. On {splits['test']['rows']:,} test rows, "
        f"it achieved MAE {row.mae:.4f}, RMSE {row.rmse:.4f} and R² {row.r2:.5f}. "
        f"Its nominal 90% residual intervals covered {row.coverage:.1%} of test targets, "
        f"with mean width {row.mean_width:.3f} index points."
    )
    tree_row = scores.set_index("model").loc["XGBoost"]
    blend_row = scores.set_index("model").loc["Blend"]
    tradeoff = (
        f"XGBoost has the lowest test MAE ({tree_row.mae:.4f}). The blend changes test RMSE "
        f"from {tree_row.rmse:.4f} to {blend_row.rmse:.4f}, while its MAE is {blend_row.mae:.4f}. "
        "This is a small observed tradeoff, not evidence of a statistically significant improvement. "
        "Orthophosphate and ammonia dominate validation permutation importance in both models. "
        "Those measurements directly enter the source formula, so this ranking is not a causal finding."
    )
    subgroup_caution = (
        "Overall interval coverage conceals substantial conditional undercoverage: the selected "
        f"model covers only {band.set_index('group').loc['Marginal', 'coverage']:.1%} of Marginal "
        f"rows and {band.set_index('group').loc['Poor', 'coverage']:.1%} of Poor rows. "
        "A global residual radius is inadequate for uniform reliability across index bands. "
        "Canada has only 46 test observations and China has none; country conclusions are limited."
    )
    candidates = pd.read_csv(out / "validation_candidates.csv")
    warning = (
        "The target is the dataset authors' derived row-level CCME_Values, not independently "
        "observed water quality. All eight target components are inputs. A deterministic formula "
        f"reproduces {formula['fraction_matching_at_1e_7']:.1%} of all source labels within 1e-7 "
        f"(maximum error {formula['max_absolute_error']:.2e}). High ML accuracy therefore measures "
        "formula emulation, not discovery of chemical mechanisms or safety."
    )
    methodology = (
        f"Remove {audit['exact_duplicates_removed']:,} exact duplicate rows. Keep pollution extremes. "
        "Flag negative concentrations, pH outside [0,14], nonfinite readings and temperature "
        "outside [-5,50] °C; the temperature bound is a conservative review rule, not a universal "
        "physical law. Replace flagged inputs with missing values and add eight quality flags. "
        "Fit median imputation on training only; MLP and Ridge additionally use asinh and "
        "standard scaling fitted on training. Split by date before fitting any transform. "
        f"Use the same seeded uniform sample of {splits['training_rows_used']:,} training rows "
        f"for every model and {splits['tuning_rows_used']:,} validation rows for candidate selection "
        "and early stopping. The full validation partition selects the model family and blend; "
        "2020 data calibrate residual intervals; the complete 2021–2023 partition is evaluated "
        "after model selection. Neither calibration nor test rows are used for training."
    )
    limitations = [
        (
            "The source adaptation checks all objectives as upper limits, including dissolved oxygen "
            "and pH, and divides scope/frequency/excursions by 12 although there are eight chemistry "
            "variables. It sets F2=F1 for one row. This differs from official site-period CCME WQI. "
            "The denominator is inferred from exact agreement with the delivered file; the notebook "
            "uses len(df.columns), so its result depends on the frame at execution time."
        ),
        (
            "The supplied data are already harmonized and imputed. Original missingness, censoring, "
            "detection limits and every unit conversion cannot be reconstructed. USA temperatures "
            "such as 63 under a Celsius header suggest unresolved unit questions; no Fahrenheit "
            "conversion is guessed. Quality flags cannot reverse upstream processing or repair targets."
        ),
        (
            f"England contributes {audit['country_counts']['England'] / audit['raw_rows']:.1%} of raw rows. "
            "Only countries actually present in the future test period have country scores. Their "
            "performance does not establish generalization to absent countries or new regions."
        ),
        (
            f"This is a time-separated same-time-input benchmark, not a forecast: "
            f"{splits['test_rows_at_sites_seen_before_2019']:.1%} of test rows have a Country+Area "
            "identifier seen in the historical training partition. Area is a coarse proxy; China "
            "has one unique Area. No claim of unseen-site generalization is made."
        ),
        (
            "The nominal 90% residual interval uses a finite-sample corrected calibration quantile. "
            "Temporal dependence and distribution shift violate exchangeability; coverage must be "
            "read empirically. It is a constant-radius interval, not learned chemical uncertainty, "
            "and has not been validated for artificially missing sensors."
        ),
        (
            "A bounded two-candidate search per learned family and one seed are reported. No "
            "statistical significance, exhaustive tuning or seed-stability claim is made. Permutation "
            "importance and SHAP explain predictions of a derived index; correlated inputs and "
            "upstream imputation prevent causal interpretation."
        ),
    ]
    conclusion = (
        "Use the transparent source formula when the task is to reproduce this exact index "
        "from all eight complete inputs. XGBoost and the MLP are useful comparative learning "
        "experiments here. A meaningful next study needs independent outcomes or an official "
        "site-window index with documented local objectives, and withheld sensors or genuinely "
        "future measurements. Do not treat the released labels as a drinking-water decision."
    )
    sources = [
        (
            "Dataset and attribution · Karim et al. · Figshare v2 · CC BY 4.0",
            "https://doi.org/10.6084/m9.figshare.27800394.v2",
        ),
        (
            "CCME WQI 2017 manual · official index definition",
            "https://ccme.ca/en/res/wqimanualen.pdf",
        ),
        ("XGBoost · Chen & Guestrin (2016)", "https://doi.org/10.1145/2939672.2939785"),
        (
            "Deep Learning · Goodfellow, Bengio & Courville (2016), chapter 6",
            "https://www.deeplearningbook.org/contents/mlp.html",
        ),
        ("Conformal prediction · Angelopoulos & Bates (2021)", "https://arxiv.org/abs/2107.07511"),
    ]
    md = (
        f"""# Water quality prediction: XGBoost and neural networks

## Result

{findings}

{tradeoff}

**Interpretation:** {warning}

## Data and target

{audit["raw_rows"]:,} source rows, 14 columns, five countries, 1940–2023. Eight inputs:
ammonia, BOD, dissolved oxygen, orthophosphate, pH, temperature, nitrogen and nitrate.
Concentrations use mg/L; pH is unitless
and temperature is labelled °C. Turbidity, conductivity and dissolved solids are absent.
Only continuous `CCME_Values` is predicted; `CCME_WQI` and all metadata are excluded from inputs.

The original notebooks and source workbooks are preserved under `notebooks/reference/`
and `data/reference/`. See [data card](../docs/DATA.md) for the formula and audit.

## Protocol

{methodology}

{markdown_table(split_frame, ["partition", "rows", "start", "end"])}

These are available partition sizes; only the training and tuning samples are capped.
The exact configuration, data hash, versions and execution time are in `run_manifest.json`.
Historical sites may recur; split by time is not split by site. Labels still depend on same-row inputs.

## Model specification and validation search

All learned models receive eight measurements and eight missing/invalid flags.
XGBoost: histogram trees, learning rate 0.05, at most 1,000 rounds, depths 4/6,
minimum child weight 5, row subsampling 0.9, L2 penalty 5, 40-round early stopping.
MLP: 64→32 or 128→64 ReLU hidden layers, dropout 0.1, one linear regression output,
AdamW (learning rate 0.001, weight decay 0.0001), MSE on targets divided by 100,
batch size 2,048, at most 60 epochs and patience 10. Candidate checkpoints minimize
validation RMSE. Predictions are clipped to [0,100]. The selected MLP reached the
epoch budget; this does not establish full convergence. Ridge uses alpha 10 on the
same transformed inputs; the constant baseline predicts the training mean.

{markdown_table(candidates, ["family", "architecture", "mae", "rmse", "best_step", "seconds"])}

Candidate scores use the sampled validation set; the final comparison below reports
full-validation RMSE. The blend searches 21 XGBoost weights from 0 to 1 by 0.05.

## Full chronological test results

{markdown_table(scores, ["model", "mae", "rmse", "r2", "coverage", "mean_width", "validation_rmse"])}

MAE/RMSE are index points, lower is better. R² is relative to the test-target mean;
negative values are possible. Coverage and mean width describe nominal 90% intervals.
Blend weight on XGBoost: **{metadata["blend_xgboost_weight"]:.2f}**, selected on validation.
The exact source formula is an audit reference, not a trained competitor: test MAE
{audit["formula_test_metrics"]["mae"]:.3e}.

## Selected-model subgroup results

{markdown_table(country, ["group", "rows", "mae", "rmse", "coverage"])}

{markdown_table(band, ["group", "rows", "mae", "rmse", "coverage"])}

Index bands are descriptive slices of the released labels, not a separate classification task.
Group R² can be unstable for near-constant targets; consult raw subgroup CSV for full scores.

{subgroup_caution}

## Interpretation

![Model comparison](figures/model_comparison.png)

![Permutation importance](figures/feature_importance.png)

![Native TreeSHAP](figures/shap_importance.png)

Permutation importance measures the MAE increase when a measurement and its missing flag
are shuffled together on {provenance["config"]["explanation_rows"]:,} validation rows, with three repeats.
Native TreeSHAP uses the selected boosting rounds and explains un-clipped tree outputs.
It includes missing-flag contributions separately in `shap_importance.csv`.

## Limitations

"""
        + "\n\n".join(f"- {item}" for item in limitations)
        + f"""

## Recommendation

{conclusion}

## References

"""
        + "\n".join(f"- [{title}]({url})" for title, url in sources)
        + "\n"
    )
    (out / "report.md").write_text(md)
    cards = "".join(
        f'<div class="card"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'
        for label, value in [
            ("Source records", f"{audit['raw_rows']:,}"),
            ("Selected on validation", selected),
            ("Test RMSE · index points", f"{row.rmse:.3f}"),
            ("Test R²", f"{row.r2:.5f}"),
        ]
    )
    figures = [
        (
            "data_overview.png",
            "01 / Know the data",
            "Unequal representation and a derived target shape this benchmark.",
        ),
        (
            "model_comparison.png",
            "02 / Compare on future observations",
            "Every model uses the same training rows; 2021–2023 is held out.",
        ),
        (
            "actual_vs_predicted.png",
            "03 / Inspect the predictions",
            "Density plots show a seeded sample of test observations. The dashed line is perfect reproduction.",
        ),
        (
            "feature_importance.png",
            "04 / Explain the model behavior",
            "Shuffling validation measurements tests model reliance, measured as the increase in MAE.",
        ),
        (
            "shap_importance.png",
            "05 / Attribute tree predictions",
            "Exact native TreeSHAP contributions in index units. Attribution does not imply causation.",
        ),
        (
            "learning_curves.png",
            "06 / Check convergence",
            "Validation curves govern early stopping; test results do not influence training.",
        ),
        (
            "country_performance.png",
            "07 / Inspect distribution shift",
            "Overall accuracy can conceal country differences and interval undercoverage.",
        ),
    ]
    panels = "".join(
        f'<section><h2>{title}</h2><p>{caption}</p><img src="figures/{name}" alt="{escape(title)}" loading="lazy"></section>'
        for name, title, caption in figures
    )
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Water Quality · Research Report</title><style>
:root{--ink:#163640;--muted:#526f78;--teal:#087f8c;--paper:#f4f8f8;--line:#dce7e8}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 system-ui,-apple-system,sans-serif}
header{padding:72px max(6vw,24px) 64px;background:#113e48;color:#fff;border-top:6px solid #65d4c3}
header .eyebrow{letter-spacing:.2em;text-transform:uppercase;font-size:12px;color:#9fdbd4}h1{font-size:clamp(38px,5.5vw,70px);line-height:1.08;letter-spacing:-.045em;max-width:950px;margin:22px 0}
header p{max-width:720px;color:#c4dedf;font-size:18px}main{max-width:1160px;margin:auto;padding:32px 24px 64px}
h2{font-size:25px;letter-spacing:-.025em;margin:0 0 12px}h3{margin-top:28px}p{color:var(--muted)}a{color:var(--teal)}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:0 0 28px}.card,section{background:white;border:1px solid var(--line);border-radius:14px}
.card{padding:22px}.card span{font-size:12px;color:var(--muted);display:block}.card strong{font-size:28px;letter-spacing:-.04em;display:block;margin-top:8px}
section{padding:30px;margin:22px 0}section img{display:block;max-width:100%;height:auto;margin:20px auto 0}.note{border-left:4px solid #df9862;background:#fff9f1}
.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}th{text-align:left;color:var(--teal);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
td,th{padding:12px 14px;border-bottom:1px solid var(--line);white-space:nowrap}li{margin:12px 0;color:var(--muted)}footer{padding-top:20px;color:var(--muted);font-size:13px}
@media(max-width:700px){.cards{grid-template-columns:repeat(2,1fr)}section{padding:20px}.card{padding:16px}.card strong{font-size:23px}header{padding-top:45px}}
@media print{header{padding:20px;color:var(--ink);background:white}section{break-inside:avoid}body{background:white}.cards{grid-template-columns:repeat(4,1fr)}}
</style></head><body><header><div class="eyebrow">Project 18 · Applied machine learning research</div>
<h1>Water quality,<br>with evidence.</h1><p>Gradient boosting and neural networks for emulating a published surface-water index. A chronological comparison with an explicit target audit.</p></header><main>
"""
    table = scores[["model", "mae", "rmse", "r2", "coverage", "mean_width"]]
    html += f'<div class="cards">{cards}</div><section class="note"><h2>Read the target before the scores</h2><p>{escape(warning)}</p></section>'
    html += f'<section><h2>The measured result</h2><p>{escape(findings)}</p><p>{escape(tradeoff)}</p><div class="tablewrap">{table.to_html(index=False, float_format=lambda v: f"{v:.4f}", border=0)}</div></section>'
    html += f'<section><h2>A fair, reproducible comparison</h2><p>{escape(methodology)}</p><div class="tablewrap">{split_frame[["partition", "rows", "start", "end"]].to_html(index=False, border=0)}</div></section>'
    html += panels
    html += f'<section class="note"><h2>Check reliability within groups</h2><p>{escape(subgroup_caution)}</p><div class="tablewrap">{country[["group", "rows", "mae", "rmse", "coverage"]].to_html(index=False, float_format=lambda v: f"{v:.4f}", border=0)}</div></section>'
    html += (
        "<section><h2>Limits of the evidence</h2><ul>"
        + "".join(f"<li>{escape(item)}</li>" for item in limitations)
        + "</ul></section>"
    )
    html += f'<section><h2>What this means in practice</h2><p>{escape(conclusion)}</p><p><a href="report.md">Full technical report</a> · <a href="metrics.csv">Metrics CSV</a> · <a href="run_manifest.json">Reproducibility manifest</a></p></section>'
    html += (
        "<footer>Sources: "
        + " · ".join(f'<a href="{url}">{escape(title)}</a>' for title, url in sources)
        + "</footer></main></body></html>"
    )
    (out / "report.html").write_text(html)


def rebuild_report(output="reports", artifacts="artifacts"):
    """Regenerate figures and prose without refitting or changing any result."""
    out = Path(output)

    def read_json(name):
        import json

        return json.loads((out / name).read_text())

    import json

    metadata = json.loads((Path(artifacts) / "metadata.json").read_text())
    build_report(
        out,
        read_json("data_audit.json"),
        read_json("split_manifest.json"),
        pd.read_csv(out / "metrics.csv"),
        pd.read_csv(out / "subgroup_metrics.csv"),
        pd.read_csv(out / "permutation_importance.csv"),
        pd.read_csv(out / "shap_importance.csv"),
        read_json("training_history.json"),
        pd.read_csv(out / "predictions_test.csv.gz"),
        metadata,
        read_json("run_manifest.json"),
    )
