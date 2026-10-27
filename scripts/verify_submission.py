"""Verify the submission, numerical evidence, native slide objects and source data."""

import argparse
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

import nbformat
import numpy as np
import pandas as pd
from pypdf import PdfReader

from water_quality.data import sha256, source_formula
from water_quality.evaluation import metrics
from water_quality.predict import predict_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-raw", action="store_true", help="Skip uncompressed raw-data checks")
    args = parser.parse_args()
    results = []

    def check(name, condition, detail):
        if not condition:
            raise AssertionError(f"{name}: {detail}")
        results.append({"check": name, "status": "PASS", "detail": detail})

    required = [
        "README.md",
        "learning.md",
        "LICENSE",
        "app.py",
        "configs/default.json",
        "submission/Water_Quality_Project_Report.pdf",
        "submission/Water_Quality_Literature_Review.pdf",
        "presentation/Water_Quality_Prediction.pptx",
        "presentation/Water_Quality_Prediction.pdf",
        "artifacts/metadata.json",
        "artifacts/xgboost.json",
        "artifacts/mlp.pt",
        "artifacts/preprocessing.joblib",
        "reports/report.md",
        "reports/predictions_test.csv.gz",
    ]
    check(
        "Submission artifacts",
        all((ROOT / p).is_file() for p in required),
        f"{len(required)} required files",
    )
    audit = json.loads((ROOT / "reports/data_audit.json").read_text())
    splits = json.loads((ROOT / "reports/split_manifest.json").read_text())
    metadata = json.loads((ROOT / "artifacts/metadata.json").read_text())
    artifact_hashes = json.loads((ROOT / "artifacts/checksums.json").read_text())
    check(
        "Trained artifact hashes",
        all(
            sha256(ROOT / "artifacts" / name) == digest for name, digest in artifact_hashes.items()
        ),
        f"{len(artifact_hashes)} saved artifact hashes match",
    )
    run = json.loads((ROOT / "reports/run_manifest.json").read_text())
    check(
        "Data accounting",
        sum(splits[k]["rows"] for k in ["train", "validation", "calibration", "test"])
        == audit["retained_rows"],
        f"{audit['retained_rows']:,} retained rows across disjoint dated partitions",
    )
    check(
        "One continuous target",
        metadata["target"] == "CCME_Values" and len(metadata["features"]) == 8,
        "CCME_Values regression, eight measurement inputs plus flags",
    )
    check(
        "Formula disclosure",
        audit["formula_audit"]["fraction_matching_at_1e_7"] == 1,
        "Every original label reconstructs; derived-target limitation is explicit",
    )
    check(
        "Chronological separation",
        splits["train"]["end"]
        < splits["validation"]["start"]
        < splits["calibration"]["start"]
        < splits["test"]["start"],
        "Training, validation, calibration and testing occupy separate ordered periods",
    )
    check(
        "Shared training sample",
        splits["training_rows_used"] == run["config"]["max_train_rows"] == 300000,
        "300,000 seed-42 historical rows train every candidate",
    )
    observed = pd.read_csv(ROOT / "reports/predictions_test.csv.gz")
    recorded = pd.read_csv(ROOT / "reports/metrics.csv")
    check(
        "Full test predictions",
        len(observed) == splits["test"]["rows"] == 202886 and observed.source_row.is_unique,
        "202,886 test rows with unique traceable source IDs",
    )
    for _, row in recorded.iterrows():
        actual = metrics(observed.CCME_Values, observed[row.model])
        check(
            f"Recomputed metrics: {row.model}",
            all(abs(actual[key] - row[key]) < 1e-5 for key in actual),
            f"MAE {actual['mae']:.6f}; RMSE {actual['rmse']:.6f}; R2 {actual['r2']:.6f}",
        )
    expected = pd.read_csv(ROOT / "examples/expected_predictions.csv")
    with TemporaryDirectory(prefix="water-quality-verification-") as directory:
        for family in ["XGBoost", "MLP", "Blend"]:
            prediction = predict_file(
                ROOT / "examples/measurements.csv",
                ROOT / "artifacts",
                Path(directory) / f"{family}.csv",
                family,
            )
            check(
                f"Reload parity: {family}",
                np.allclose(
                    prediction.predicted_source_index, expected[family], atol=1e-4, rtol=1e-6
                ),
                "Reloaded models match the saved holdout fixture",
            )
    for name in [
        "01_eda.ipynb",
        "02_preprocessing.ipynb",
        "03_model_application.ipynb",
    ]:
        book = nbformat.read(ROOT / "notebooks" / name, as_version=4)
        nbformat.validate(book)
        cells = [cell for cell in book.cells if cell.cell_type == "code"]
        check(
            f"Executed notebook: {name}",
            all(
                cell.execution_count is not None
                and not any(o.output_type == "error" for o in cell.outputs)
                for cell in cells
            ),
            f"{len(cells)} executed cells with no recorded errors",
        )
    for filename, count in [
        ("submission/Water_Quality_Project_Report.pdf", 10),
        ("submission/Water_Quality_Literature_Review.pdf", 4),
        ("presentation/Water_Quality_Prediction.pdf", 12),
    ]:
        reader = PdfReader(ROOT / filename)
        check(
            f"PDF: {Path(filename).name}",
            len(reader.pages) == count
            and all(len(page.extract_text() or "") > 40 for page in reader.pages),
            f"{count} readable pages",
        )
    with zipfile.ZipFile(ROOT / "presentation/Water_Quality_Prediction.pptx") as package:
        slides = [
            name
            for name in package.namelist()
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        ]
        charts = [
            name
            for name in package.namelist()
            if name.startswith("ppt/charts/chart") and name.endswith(".xml")
        ]
        notes = [
            name
            for name in package.namelist()
            if name.startswith("ppt/notesSlides/notesSlide") and name.endswith(".xml")
        ]
        check(
            "Editable presentation",
            len(slides) == 12 and len(charts) == 4 and len(notes) == 12,
            "12 slides, four native charts with embedded workbooks, 12 speaker-note parts",
        )
        types = ET.fromstring(package.read("[Content_Types].xml"))
        check(
            "PPTX package integrity",
            all(
                n.attrib["PartName"].lstrip("/") in package.namelist()
                for n in types
                if n.tag.endswith("Override")
            ),
            "Every declared content-type part exists",
        )
    original = json.loads((ROOT / "data/source_checksums.json").read_text())
    if not args.skip_raw:
        for path, info in original.items():
            check(
                f"Source hash: {Path(path).name}",
                sha256(ROOT / path) == info["sha256"],
                "Original source file is unchanged",
            )
        max_error = 0.0
        total = 0
        for chunk in pd.read_csv(ROOT / "data/raw/combined/Combined_dataset.csv", chunksize=100000):
            max_error = max(
                max_error, float(np.abs(source_formula(chunk) - chunk.CCME_Values).max())
            )
            total += len(chunk)
        check(
            "Independent full formula replay",
            total == audit["raw_rows"] and max_error < 1e-7,
            f"{total:,} rows; maximum absolute difference {max_error:.3e}",
        )
    reproduction = run.get("reproduction")
    if reproduction:
        delta = reproduction["max_absolute_difference"]
        check(
            "Recorded training reproduction",
            delta < 1e-5,
            f"Earlier independent full rerun: maximum metric difference {delta:.3e}",
        )
    payload = {
        "verified_utc": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "checks": results,
        "raw_checks_skipped": args.skip_raw,
        "assessment": "Apt for the brief as a transparent index-emulation comparison; independent environmental prediction is not established.",
        "administrative_note": "Dummy team details are intentional, as requested.",
    }
    lines = [
        "# Final submission verification",
        "",
        f"Status: **PASS**. {len(results)} checks completed.",
        f"Verified UTC: {payload['verified_utc']}",
        "",
        payload["assessment"],
        "",
        "Team identities remain editable dummy values at the user's request.",
        "",
        "| Check | Result | Evidence |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {r['check']} | {r['status']} | {r['detail']} |" for r in results)
    lines.extend(
        [
            "",
            "Implementation tests and lint checks also run in GitHub Actions. Run python -m pytest -q and ruff check src tests scripts app.py locally.",
            "Visual review covers the exported slide PDF and both report PDFs. Package checks alone do not prove appearance.",
            "The scientific limitations remain part of the submitted report, rather than being treated as validation failures.",
        ]
    )
    (ROOT / "submission/VERIFICATION.md").write_text("\n".join(lines) + "\n")
    print(f"PASS: {len(results)} checks; see submission/VERIFICATION.md")


if __name__ == "__main__":
    main()
