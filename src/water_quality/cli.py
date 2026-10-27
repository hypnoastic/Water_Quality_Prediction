"""Command line entry points; invoke from the repository root."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Water quality index emulation research")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Audit, train, evaluate, explain, and report")
    train.add_argument("--config", default="configs/default.json")
    audit = sub.add_parser("audit", help="Audit the source data without training")
    audit.add_argument("--data", default="data/raw/combined/Combined_dataset.csv")
    audit.add_argument("--output", default="reports/data_audit.json")
    pred = sub.add_parser("predict", help="Predict source index from a measurement CSV")
    pred.add_argument("--input", required=True)
    pred.add_argument("--artifacts", default="artifacts")
    pred.add_argument("--output", default="reports/example_predictions.csv")
    pred.add_argument("--model", choices=["XGBoost", "MLP", "Blend"])
    report = sub.add_parser("report", help="Rebuild reports from an existing experiment")
    report.add_argument("--output", default="reports")
    report.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    if args.command == "train":
        from .pipeline import run

        run(json.loads(Path(args.config).read_text()))
    elif args.command == "audit":
        from .data import prepare, read_dataset
        from .pipeline import dump_json, table_audit

        frame = read_dataset(args.data)
        result = table_audit(frame)
        _, removed = prepare(frame)
        result.update(removed)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        dump_json(args.output, result)
        print(json.dumps(result, indent=2))
    elif args.command == "report":
        from .reporting import rebuild_report

        rebuild_report(args.output, args.artifacts)
        print(f"Rebuilt {args.output}/report.html and report.md")
    else:
        from .predict import predict_file

        result = predict_file(args.input, args.artifacts, args.output, args.model)
        print(
            result[["predicted_source_index", "lower_90", "upper_90", "model"]].to_string(
                index=False
            )
        )
        print(f"Saved {args.output}. Research index emulation; no water-safety determination.")


if __name__ == "__main__":
    main()
