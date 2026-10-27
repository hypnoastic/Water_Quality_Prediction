"""Local presentation demo. Run: streamlit run app.py"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st

from water_quality.data import FEATURES, SHORT_NAMES
from water_quality.predict import predict_file

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Water Quality Prediction", page_icon="💧", layout="wide")
st.title("Water Quality Prediction")
st.caption("Project 18 · XGBoost and a PyTorch neural network")
st.info(
    "This demonstration estimates the dataset authors' calculated water-quality index. "
    "It is an index-emulation experiment. The score does not certify drinking-water safety."
)
metadata = json.loads((ROOT / "artifacts/metadata.json").read_text())
model = st.selectbox("Trained model", ["Blend", "XGBoost", "MLP"])
st.caption("Blend = 90% XGBoost + 10% MLP, selected using 2019 validation RMSE.")


def infer(frame, chosen):
    with TemporaryDirectory(prefix="water-quality-demo-") as directory:
        source, output = Path(directory) / "input.csv", Path(directory) / "output.csv"
        frame.to_csv(source, index=False)
        return predict_file(source, ROOT / "artifacts", output, chosen)


single, batch, evidence = st.tabs(["One sample", "CSV prediction", "Experiment results"])
with single:
    st.subheader("Enter the sample measurements")
    default = pd.read_csv(ROOT / "examples/measurements.csv").iloc[0]
    with st.form("measurement_form"):
        columns = st.columns(2)
        values = {}
        for i, (feature, label) in enumerate(zip(FEATURES, SHORT_NAMES)):
            unit = "pH units" if i == 4 else "°C" if i == 5 else "mg/L"
            with columns[i % 2]:
                values[feature] = st.number_input(
                    f"{label} ({unit})", value=float(default[feature]), format="%.4f", key=feature
                )
        submitted = st.form_submit_button("Predict water-quality index", type="primary")
    if submitted:
        try:
            result = infer(pd.DataFrame([values]), model)
            record = result.iloc[0]
            a, b, c = st.columns(3)
            a.metric("Predicted source index", f"{record.predicted_source_index:.2f} / 100")
            b.metric("Nominal 90% interval", f"{record.lower_90:.2f} – {record.upper_90:.2f}")
            c.metric("Measurements needing review", int(record.flagged_or_missing_measurements))
            if record.input_review_required:
                st.warning("Some readings were flagged and imputed. Review the inputs and units.")
            st.caption(
                "Intervals use 2020 held-out residuals. Coverage varies by period and group, "
                "and has not been validated for new missing-sensor patterns."
            )
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
with batch:
    st.subheader("Predict several samples")
    st.write("Upload a CSV using the original measurement headers and units.")
    st.download_button(
        "Download input template",
        (ROOT / "examples/measurements.csv").read_bytes(),
        file_name="measurements.csv",
        mime="text/csv",
    )
    upload = st.file_uploader("Measurement CSV", type=["csv"], max_upload_size=10)
    if upload is not None and st.button("Predict CSV", type="primary"):
        try:
            frame = pd.read_csv(upload)
            if len(frame) > 10000:
                raise ValueError("Use the CLI for files larger than 10,000 samples.")
            result = infer(frame, model)
            st.dataframe(result, hide_index=True)
            st.download_button(
                "Download predictions",
                result.to_csv(index=False),
                file_name="water_quality_predictions.csv",
                mime="text/csv",
            )
        except (ValueError, KeyError, pd.errors.ParserError) as exc:
            st.error(str(exc))
with evidence:
    st.subheader("Chronological test results")
    st.write("300,000 historical training rows. Full test: 202,886 observations from 2021–2023.")
    metrics = pd.read_csv(ROOT / "reports/metrics.csv")
    st.dataframe(metrics[["model", "mae", "rmse", "r2", "coverage"]], hide_index=True)
    st.image(str(ROOT / "reports/figures/feature_importance.png"))
    st.warning(
        "The source formula reconstructs every released target. The selected blend's 90% "
        "intervals cover 89.3% overall, but only 45.2% of Marginal rows."
    )
    st.write("See reports/report.md for methods, source attribution and subgroup results.")
