"""Reload frozen models; prediction uses the same rules and fitted transforms."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor

from .data import quality_features
from .evaluation import intervals
from .models import MLP, predict_mlp


def predict_file(input_path, artifact_dir, output_path, model_name=None):
    directory = Path(artifact_dir)
    metadata = json.loads((directory / "metadata.json").read_text())
    chosen = model_name or metadata["selected_model"]
    if chosen not in ["XGBoost", "MLP", "Blend"]:
        raise ValueError("Prediction model must be XGBoost, MLP or Blend")
    raw = pd.read_csv(input_path)
    if raw.empty:
        raise ValueError("Input contains no measurement rows")
    x = quality_features(raw)
    if x.iloc[:, :8].isna().all(axis=1).any():
        raise ValueError(
            "Each row needs at least one valid measurement; a fully imputed result is unsupported"
        )
    # Only load artifacts produced locally by this project. joblib is pickle based.
    pre = joblib.load(directory / "preprocessing.joblib")
    if chosen in ["XGBoost", "Blend"]:
        tree = XGBRegressor()
        tree.load_model(directory / "xgboost.json")
        px = np.clip(tree.predict(pre["tree"].transform(x)), 0, 100)
    if chosen in ["MLP", "Blend"]:
        neural = MLP(16, metadata["mlp_widths"], metadata["dropout"])
        neural.load_state_dict(
            torch.load(directory / "mlp.pt", map_location="cpu", weights_only=True)
        )
        pn = predict_mlp(neural, pre["neural"].transform(x).astype("float32"))
    if chosen == "Blend":
        weight = metadata["blend_xgboost_weight"]
        prediction = weight * px + (1 - weight) * pn
    else:
        prediction = px if chosen == "XGBoost" else pn
    lower, upper = intervals(prediction, metadata["radii"][chosen])
    result = raw.copy()
    result["predicted_source_index"] = prediction
    result["lower_90"] = lower
    result["upper_90"] = upper
    result["model"] = chosen
    result["flagged_or_missing_measurements"] = x.iloc[:, 8:].sum(axis=1).astype(int).to_numpy()
    result["input_review_required"] = result.flagged_or_missing_measurements > 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result
