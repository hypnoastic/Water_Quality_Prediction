"""Build the academic report and focused literature review from measured artifacts."""

import json
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission"
INK, TEAL, MUTED = (
    colors.HexColor("#183c3e"),
    colors.HexColor("#087f8c"),
    colors.HexColor("#52686d"),
)
for name, filename in [("Body", "Arial.ttf"), ("BodyBold", "Arial Bold.ttf")]:
    candidates = [
        Path("/System/Library/Fonts/Supplemental") / filename,
        Path("/usr/share/fonts/truetype/dejavu")
        / ("DejaVuSans-Bold.ttf" if "Bold" in name else "DejaVuSans.ttf"),
    ]
    found = next((p for p in candidates if p.exists()), None)
    if found:
        pdfmetrics.registerFont(TTFont(name, str(found)))
FONT = "Body" if "Body" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
BOLD = "BodyBold" if "BodyBold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"
STYLES = getSampleStyleSheet()
STYLES.add(
    ParagraphStyle(
        "Copy",
        fontName=FONT,
        fontSize=10.8,
        leading=15.3,
        textColor=INK,
        spaceAfter=10,
        alignment=TA_LEFT,
    )
)
STYLES.add(
    ParagraphStyle("TitleW", fontName=BOLD, fontSize=29, leading=35, textColor=INK, spaceAfter=20)
)
STYLES.add(
    ParagraphStyle("HeadW", fontName=BOLD, fontSize=19, leading=24, textColor=TEAL, spaceAfter=15)
)
STYLES.add(
    ParagraphStyle("SubW", fontName=BOLD, fontSize=12, leading=17, textColor=INK, spaceAfter=8)
)
STYLES.add(
    ParagraphStyle("SmallW", fontName=FONT, fontSize=8.2, leading=11, textColor=MUTED, spaceAfter=8)
)


def p(text, style="Copy"):
    return Paragraph(text, STYLES[style])


def table(headers, rows, widths=None):
    data = [[p(str(v), "SmallW") for v in headers]]
    data += [[p(escape(str(v)), "SmallW") for v in row] for row in rows]
    obj = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    obj.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f1ef")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, TEAL),
                ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#d7e4e2")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return obj


def figure(name, width=17 * cm):
    image = Image(str(ROOT / "reports/figures" / name))
    image.drawHeight *= width / image.drawWidth
    image.drawWidth = width
    return image


def page_number(canvas, doc):
    canvas.setStrokeColor(colors.HexColor("#d7e4e2"))
    canvas.line(2 * cm, 1.7 * cm, 19 * cm, 1.7 * cm)
    canvas.setFont(FONT, 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.15 * cm, "Project 18 | Water Quality Prediction")
    canvas.drawRightString(19 * cm, 1.15 * cm, str(doc.page))


def build(path, pages):
    story = []
    for i, content in enumerate(pages):
        if i:
            story.append(PageBreak())
        story.extend(content)
    document = SimpleDocTemplate(
        str(path),
        pagesize=(21 * cm, 29.7 * cm),
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2.2 * cm,
        title=path.stem.replace("_", " "),
        author="Project 18 Team",
    )
    document.build(story, onFirstPage=page_number, onLaterPages=page_number)


REFERENCES = [
    "[1] Karim et al. A Comprehensive Surface Water Quality Monitoring Dataset (1940-2023), Figshare v2, 2025. https://doi.org/10.6084/m9.figshare.27800394.v2. CC BY 4.0.",
    "[2] Canadian Council of Ministers of the Environment. CCME Water Quality Index User's Manual, 2017 update. https://ccme.ca/en/res/wqimanualen.pdf.",
    "[3] Breiman, L. Random Forests. Machine Learning 45, 5-32, 2001. https://doi.org/10.1023/A:1010933404324.",
    "[4] Chen, T., and Guestrin, C. XGBoost: A Scalable Tree Boosting System. KDD, 2016. https://doi.org/10.1145/2939672.2939785.",
    "[5] Goodfellow, I., Bengio, Y., and Courville, A. Deep Learning. MIT Press, 2016. Chapter 6. https://www.deeplearningbook.org/contents/mlp.html.",
    "[6] Grinsztajn, L., Oyallon, E., and Varoquaux, G. Why do tree-based models still outperform deep learning on tabular data? 2022. https://arxiv.org/abs/2207.08815.",
    "[7] Srivastava, N. et al. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR 15(56), 1929-1958, 2014. https://jmlr.org/papers/v15/srivastava14a.html.",
    "[8] Angelopoulos, A. N., and Bates, S. A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. 2021, revised 2022. https://arxiv.org/abs/2107.07511.",
]


def main():
    OUT.mkdir(exist_ok=True)
    team = json.loads((OUT / "team.json").read_text())
    audit = json.loads((ROOT / "reports/data_audit.json").read_text())
    splits = json.loads((ROOT / "reports/split_manifest.json").read_text())
    metrics = pd.read_csv(ROOT / "reports/metrics.csv")
    candidates = pd.read_csv(ROOT / "reports/validation_candidates.csv")
    groups = pd.read_csv(ROOT / "reports/subgroup_metrics.csv")
    member_rows = [[m["name"], m["enrollment"]] for m in team["members"]]
    identity = (
        f"{escape(team['team_id'])} / {escape(team['section'])}<br/>{escape(team['institution'])}"
    )
    cover = [
        Spacer(1, 1.3 * cm),
        p("ACADEMIC PROJECT REPORT", "SmallW"),
        p("Water Quality Prediction", "TitleW"),
        p("Gradient Boosting and Deep Neural Networks", "HeadW"),
        p("A chronological comparison on the source-defined surface-water index"),
        Spacer(1, 0.8 * cm),
        p(identity),
        table(["Team member", "Enrollment number"], member_rows, [9 * cm, 8 * cm]),
        Spacer(1, 0.8 * cm),
        p("Project 18 | October 2026", "SubW"),
        p(
            "Team information intentionally uses dummy values at the user's request. Replace submission/team.json before adding final institutional details.",
            "SmallW",
        ),
        p("Repository: https://github.com/hypnoastic/Water_Quality_Prediction", "SmallW"),
    ]
    abstract = [
        p("Abstract and objectives", "HeadW"),
        p(
            "This project compares XGBoost and a regularized PyTorch multilayer perceptron for regression of the released CCME_Values index. The supplied five-country dataset contains 2,827,977 rows spanning 1940-2023. We remove 9,971 exact duplicates, apply declared input-quality rules and fit all preprocessing on a shared 300,000-row historical training sample. We separate 2019 validation, 2020 residual calibration and a complete 2021-2023 test partition of 202,886 rows."
        ),
        p(
            "XGBoost obtains test MAE 0.1099 and RMSE 0.4262. The selected 128/64-unit MLP obtains MAE 0.4466 and RMSE 0.9161. Validation selects a 90% XGBoost, 10% MLP blend, which obtains RMSE 0.4222 and R-squared 0.99861, with worse MAE than XGBoost. A source-formula audit reconstructs every released label within 1e-7. Consequently, the experiment establishes index emulation rather than independent environmental prediction."
        ),
        p("Research question", "SubW"),
        p(
            "How do a strong boosted-tree model and a small neural network compare when reproducing one documented continuous index under chronological evaluation?"
        ),
        p("Objectives", "SubW"),
        p(
            "Define one target and trace its construction. Audit measurements, units and prior imputation. Compare both model families against mean and Ridge baselines. Explain model reliance and examine uncertainty across countries and index bands."
        ),
        p(
            "Keywords: water quality, XGBoost, MLP, regression, temporal evaluation, derived target, interpretability.",
            "SmallW",
        ),
    ]
    data = [
        p("1. Dataset and exploratory analysis", "HeadW"),
        p(
            "Karim et al.'s Figshare v2 release [1] supplies eight chemistry measurements, four metadata columns and two related target columns. The combined CSV is the single training source. Country files are alternate views and must not be appended to it."
        ),
        table(
            ["Country", "Source rows", "Distinct Area labels"],
            [
                [c, f"{n:,}", f"{audit['area_counts_by_country'][c]:,}"]
                for c, n in audit["country_counts"].items()
            ],
            [6 * cm, 5.5 * cm, 5.5 * cm],
        ),
        Spacer(1, 0.35 * cm),
        p(
            "England contributes 75.3% of rows. China has one Area label, so Area is an imperfect site identifier. Future test observations represent four countries, with only 46 Canadian rows and no Chinese rows."
        ),
        p(
            "Inputs: ammonia, biochemical oxygen demand, dissolved oxygen, orthophosphate, pH, temperature, nitrogen and nitrate. Concentrations use the supplied mg/L headers, temperature uses a Celsius header and pH is unitless. Turbidity, conductivity and dissolved solids are absent."
        ),
        figure("data_overview.png"),
        p(
            "Figure 1. Country representation and released index bands. The country-count axis is logarithmic.",
            "SmallW",
        ),
    ]
    target = [
        p("2. Target definition and formula audit", "HeadW"),
        p(
            "We predict the continuous CCME_Values field on [0,100]. CCME_WQI is a redundant class label and never enters a model. The source notebook's objectives are 0.5, 6, 14, 0.05, 8.5, 25, 20 and 45 in the feature order above. These values reproduce the authors' labels and are not asserted to be universal regulatory limits."
        ),
        p("Released formula", "SubW"),
        p(
            "failed = count(x[j] &gt; objective[j])<br/>excursion = sum(max(x[j] / objective[j] - 1, 0))<br/>F1 = F2 = 100 * failed / 12<br/>nse = excursion / 12<br/>F3 = nse / (0.01 * nse + 0.01)<br/>index = 100 - sqrt(F1<super>2</super> + F2<super>2</super> + F3<super>2</super>) / 1.732"
        ),
        p(
            "The reconstruction matches all 2,827,977 source rows within 1e-7, with maximum error 2.84e-14. The denominator of 12 is inferred from exact agreement. The original code uses len(df.columns), making its denominator depend on the frame at execution."
        ),
        p("Relationship to official CCME WQI", "SubW"),
        p(
            "The official method [2] summarizes scope, frequency and amplitude for defined variables and a sampling period, using suitable upper, lower or range objectives. The source adaptation uses individual rows, F2=F1 and upper-only comparisons, including dissolved oxygen and pH. With eight chemistry variables, a denominator of 12 is not a count of measured variables."
        ),
        p("Interpretation", "SubW"),
        p(
            "High ML scores reflect approximation of a known calculation from its own components. Temporal separation limits future-data contamination but cannot remove that structural dependence. Computing the transparent source formula is preferable when all eight complete inputs are available."
        ),
    ]
    protocol = [
        p("3. Preprocessing and experimental design", "HeadW"),
        p(
            "We parse dates strictly as day-month-year, retain original row IDs and remove exact duplicate rows. We never impute targets or invalid dates. All delivered cells are filled, but the source notebook documents upstream imputation. Original missingness and detection-limit flags cannot be reconstructed."
        ),
        p(
            "Quality rules mask negative concentrations, nonfinite readings and pH outside [0,14]. Temperature outside [-5,50] Celsius triggers a review flag. This is a declared screening range, not a universal physical limit. We do not guess Fahrenheit conversions. High pollution concentrations remain in the experiment."
        ),
        p(
            "Each reading receives a missing/invalid flag. Training medians impute masked inputs. Trees use those imputed readings and flags. MLP and Ridge additionally use asinh followed by training-fitted standard scaling. The target remains the original released index."
        ),
        table(
            ["Partition", "Dates", "Available rows", "Rows used"],
            [
                ["Train", "1940-2018", f"{splits['train']['rows']:,}", "300,000"],
                [
                    "Validation",
                    "2019",
                    f"{splits['validation']['rows']:,}",
                    "50,000 tuning / full selection",
                ],
                ["Calibration", "2020", f"{splits['calibration']['rows']:,}", "All"],
                ["Test", "2021-2023", f"{splits['test']['rows']:,}", "All"],
            ],
            [3 * cm, 3.5 * cm, 4 * cm, 6.5 * cm],
        ),
        Spacer(1, 0.4 * cm),
        p(
            "Every model shares the same seed-42 training sample. Validation chooses candidate checkpoints, model family and blend weight. Calibration only sets residual radii. Final test predictions follow frozen choices. Historical sites may recur: 85.9% of test rows have a Country+Area label seen before 2019."
        ),
    ]
    models = [
        p("4. Models and tuning", "HeadW"),
        p(
            "Mean and Ridge baselines establish reference performance. Ridge uses alpha=10 on the same transformed inputs as the MLP. XGBoost [4] provides a strong threshold-based tabular model. The MLP [5] tests learned nonlinear combinations without imposing a false sequence on feature columns."
        ),
        p("XGBoost", "SubW"),
        p(
            "Histogram trees, learning rate 0.05, depths 4 or 6, at most 1,000 estimators, minimum child weight 5, subsample 0.9, L2 penalty 5 and early stopping patience 40. The chosen depth-6 model uses 803 boosting rounds."
        ),
        p("Neural network", "SubW"),
        p(
            "Sixteen inputs comprise eight readings and eight flags. We compare hidden widths 64/32 and 128/64 with ReLU, dropout 0.1 [7] and one linear output. AdamW uses learning rate 0.001 and weight decay 0.0001. We train MSE on targets divided by 100, with batch size 2,048, maximum 60 epochs and validation patience 10. The selected 128/64 model reaches the epoch budget. Outputs are clipped to [0,100]."
        ),
        table(
            ["Family", "Candidate", "Validation RMSE", "Best round/epoch"],
            [
                [r.family, r.architecture, f"{r.rmse:.4f}", int(r.best_step)]
                for _, r in candidates.iterrows()
            ],
            [3.5 * cm, 4.5 * cm, 4.5 * cm, 4.5 * cm],
        ),
        Spacer(1, 0.35 * cm),
        p(
            "The blend searches 21 weights on full validation predictions. The selected prediction equals 0.9 times XGBoost plus 0.1 times MLP. Candidate counts are bounded and equal by family, but this is not exhaustive tuning or an equal-compute benchmark."
        ),
    ]
    results = [
        p("5. Results and model comparison", "HeadW"),
        p(
            "MAE and RMSE use index points, where lower is better. RMSE weights larger errors more strongly. R-squared compares squared error with the test-target mean and may be negative. Model selection uses full-validation RMSE, never final test scores."
        ),
        table(
            ["Model", "MAE", "RMSE", "R-squared"],
            [
                [r.model, f"{r.mae:.4f}", f"{r.rmse:.4f}", f"{r.r2:.5f}"]
                for _, r in metrics.iterrows()
            ],
            [6.5 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm],
        ),
        Spacer(1, 0.3 * cm),
        figure("actual_vs_predicted.png"),
        p(
            "Figure 2. Prediction density for a seeded test sample. Dashed lines denote exact reproduction.",
            "SmallW",
        ),
        p(
            "XGBoost wins the individual-model comparison. The blend slightly improves RMSE from 0.4262 to 0.4222, but increases MAE from 0.1099 to 0.1267. The observed change is small and one seed does not support a significance claim. The deterministic formula has test MAE approximately 1.18e-15, reinforcing the emulation interpretation."
        ),
    ]
    explanation = [
        p("6. Feature explanations", "HeadW"),
        p(
            "Permutation importance shuffles each measurement and its quality flag together on 2,000 validation rows, with three repeats. It measures the increase in validation MAE, allowing a comparison in index units. Native TreeSHAP attributes the selected XGBoost rounds without a separate SHAP package."
        ),
        figure("feature_importance.png"),
        p(
            "Figure 3. Validation permutation importance. Error bars summarize repeat variation.",
            "SmallW",
        ),
        figure("shap_importance.png", width=14.8 * cm),
        p(
            "Figure 4. Mean absolute native TreeSHAP contribution. Missing-flag contributions are retained in the CSV.",
            "SmallW",
        ),
        p(
            "Orthophosphate and ammonia dominate model reliance. Both directly enter the target formula. Correlation, harmonization and imputation affect attribution. Neither permutation importance nor SHAP establishes that changing a measurement causes a corresponding environmental outcome."
        ),
    ]
    uncertainty = [
        p("7. Uncertainty and subgroup reliability", "HeadW"),
        p(
            "Using 2020 held-out data, we take a finite-sample-corrected 90th percentile of absolute residuals [8]. Adding and subtracting this radius forms a constant-radius interval, clipped to [0,100]. Formal split-conformal guarantees depend on exchangeability. Temporal dependence and distribution shift make empirical coverage the relevant evidence here."
        ),
        table(
            ["Model", "Test coverage", "Mean width"],
            [[r.model, f"{r.coverage:.1%}", f"{r.mean_width:.4f}"] for _, r in metrics.iterrows()],
            [7 * cm, 5 * cm, 5 * cm],
        ),
        Spacer(1, 0.3 * cm),
        table(
            ["Selected blend slice", "Rows", "MAE", "Coverage"],
            [
                [r.group, f"{r.rows:,}", f"{r.mae:.4f}", f"{r.coverage:.1%}"]
                for _, r in groups[
                    (groups.model == "Blend") & (groups.group_type == "source_index_band")
                ].iterrows()
            ],
            [6 * cm, 3.5 * cm, 3.5 * cm, 4 * cm],
        ),
        Spacer(1, 0.4 * cm),
        p(
            "The blend covers 89.3% overall, but only 45.2% of Marginal rows and 51.9% of Poor rows. A global radius fails to provide uniform conditional reliability. Canada has 46 test observations with 47.8% blend coverage. China has no test rows. New missing-sensor patterns are not covered by this validation."
        ),
    ]
    conclusion = [
        p("8. Conclusion, deliverables and references", "HeadW"),
        p(
            "The project meets the brief as a documented continuous-index comparison: strong AML baseline, regularized neural comparator, chronological split, train-only preprocessing, regression metrics and feature explanations. The target audit is essential to the conclusion. The results support formula emulation, not independent environmental prediction, new-site generalization or water-safety decisions."
        ),
        p(
            "The submission includes saved models, a CLI, an interactive local demo, executed notebooks, a focused literature review, presentation files, raw-data download checksums and automated verification. Dummy team identities are the only intentionally unfinished administrative fields."
        ),
        p(
            "Future work should obtain independent outcomes or a properly defined site-window index, preserve original sampling and censoring flags, prespecify a reduced sensor panel and repeat rolling evaluation across seeds. A more elaborate neural model cannot repair a target-definition problem."
        ),
        p("References", "SubW"),
        *[p(escape(ref), "SmallW") for ref in REFERENCES],
        p(
            "The Flight_Delay_Prediction repository supplied by the user informed the submission organization only. No flight data or evaluation results enter this project. https://github.com/abhi-7-7/Flight_Delay_Prediction",
            "SmallW",
        ),
    ]
    build(
        OUT / "Water_Quality_Project_Report.pdf",
        [
            cover,
            abstract,
            data,
            target,
            protocol,
            models,
            results,
            explanation,
            uncertainty,
            conclusion,
        ],
    )

    literature = [
        [
            p("Focused Literature Review", "TitleW"),
            p("Water Quality Prediction", "HeadW"),
            p("Tree ensembles, feedforward neural networks and evaluation of a calculated index"),
            p(identity),
            table(["Team member", "Enrollment number"], member_rows, [9 * cm, 8 * cm]),
            Spacer(1, 0.7 * cm),
            p("Scope", "SubW"),
            p(
                "This is a focused review of the methods and target definition used in Project 18, rather than an exhaustive systematic review of water-quality research. The sources below connect directly to implemented choices. We separate theoretical motivation from observations produced by the submitted code."
            ),
            p(
                "The central issue is construct validity: learning a score calculated from the same predictors tests an emulator. A valid index definition and honest evaluation are necessary before comparing algorithm complexity."
            ),
            p(
                "Source access: October 2026. Dummy team information is intentional and editable in submission/team.json.",
                "SmallW",
            ),
        ],
        [
            p("1. Index definition and tree models", "HeadW"),
            p("Karim et al. dataset [1]", "SubW"),
            p(
                "The release combines eight harmonized chemistry parameters across five countries and provides both a numerical index and class bands. It enables a common regression target, but its supplied preprocessing has already filled missing values. The original observation flags and full conversion history are unavailable. We therefore preserve source files and audit the released formula rather than inventing laboratory provenance."
            ),
            p("CCME user manual [2]", "SubW"),
            p(
                "The manual defines an index through scope, frequency and amplitude against chosen objectives over a defined assessment period. Objectives and temporal aggregation are part of the target definition. The released notebook's single-row, upper-only adaptation differs from that framework. This distinction limits environmental interpretation and motivates the exact-reconstruction check."
            ),
            p("Random Forests [3]", "SubW"),
            p(
                "Breiman's forest averages randomized trees to reduce prediction variance. It provides the assignment's classical tabular-model background. We select XGBoost as the implemented AML baseline; a random forest is discussed for context and is not presented as a trained competitor."
            ),
            p("XGBoost [4]", "SubW"),
            p(
                "Chen and Guestrin describe a scalable regularized boosting system. Sequential trees model residual structure and threshold interactions. In this project, bounded tree depths and early stopping control fitting. The source formula contains explicit thresholds, making this a plausible model choice before viewing the test results."
            ),
        ],
        [
            p("2. Neural networks and reliability", "HeadW"),
            p("Deep feedforward networks [5]", "SubW"),
            p(
                "A multilayer perceptron composes learned affine transformations with nonlinear activations. This motivates a small regression network for interacting sensor measurements. Scaling improves optimization, while a linear output predicts a continuous quantity. Tabular columns do not form a temporal sequence, so an LSTM is not justified by their order."
            ),
            p("Tabular model comparisons [6]", "SubW"),
            p(
                "Grinsztajn and colleagues benchmark tree ensembles and neural models on tabular problems and analyze their differing inductive biases. Their findings motivate testing a strong tree baseline before claiming an advantage for deep learning. Their benchmark does not predict which model must win on this dataset; our results remain separate empirical evidence."
            ),
            p("Dropout [7]", "SubW"),
            p(
                "Srivastava and colleagues propose dropping units during training to reduce co-adaptation. We use dropout 0.1 with weight decay and validation stopping. This project does not include a dropout ablation, so its individual contribution cannot be isolated from the other regularizers."
            ),
            p("Conformal prediction [8]", "SubW"),
            p(
                "Angelopoulos and Bates explain uncertainty sets calibrated using held-out errors under exchangeability. We use the corrected residual quantile as a transparent interval method. Because chronology introduces dependence and shift, we assess observed coverage and avoid promising distribution-free validity for this particular holdout."
            ),
            p("Resulting research gap", "SubW"),
            p(
                "The present dataset supports a reproducible methods comparison, but independent quality outcomes and original measurement flags remain missing. The next contribution should improve target independence and evaluation design before adding network depth."
            ),
        ],
        [
            p("3. Evidence-to-method mapping", "HeadW"),
            table(
                ["Source", "Implemented decision", "Limit of inference"],
                [
                    [
                        "Dataset + CCME [1,2]",
                        "One released numerical target and exact formula audit",
                        "No claim of official site-period assessment",
                    ],
                    [
                        "XGBoost [4]",
                        "Regularized depth-4/depth-6 search",
                        "Bounded search and one seed",
                    ],
                    [
                        "Feedforward networks [5]",
                        "Small ReLU MLP, transformed inputs",
                        "No guarantee of DL superiority",
                    ],
                    [
                        "Tabular benchmark [6]",
                        "Strong classical baseline before DL",
                        "External results do not replace local evaluation",
                    ],
                    [
                        "Dropout [7]",
                        "Dropout plus AdamW and early stopping",
                        "No isolated regularizer ablation",
                    ],
                    [
                        "Conformal overview [8]",
                        "Held-out residual intervals",
                        "Temporal and subgroup coverage can fail",
                    ],
                ],
                [3.7 * cm, 6.6 * cm, 6.7 * cm],
            ),
            Spacer(1, 0.4 * cm),
            p("References", "SubW"),
            *[p(escape(ref), "SmallW") for ref in REFERENCES],
        ],
    ]
    build(OUT / "Water_Quality_Literature_Review.pdf", literature)
    print("Built the project report and literature review from measured artifacts.")


if __name__ == "__main__":
    main()
