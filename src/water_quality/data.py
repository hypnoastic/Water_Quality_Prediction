"""Schema, deterministic quality rules, source-formula audit, and time splits."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "Ammonia (mg/l)",
    "Biochemical Oxygen Demand (mg/l)",
    "Dissolved Oxygen (mg/l)",
    "Orthophosphate (mg/l)",
    "pH (ph units)",
    "Temperature (cel)",
    "Nitrogen (mg/l)",
    "Nitrate (mg/l)",
]
SHORT_NAMES = [
    "Ammonia",
    "BOD",
    "Dissolved oxygen",
    "Orthophosphate",
    "pH",
    "Temperature",
    "Nitrogen",
    "Nitrate",
]
TARGET = "CCME_Values"
OBJECTIVES = np.array([0.5, 6, 14, 0.05, 8.5, 25, 20, 45])
META = ["Country", "Area", "Waterbody Type", "Date"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_formula(frame):
    """Reconstruct released labels, including denominator 12 and upper-only tests.

    Diagnostic only: this intentionally preserves source methodological issues.
    It is NOT an implementation of the official site/time-window CCME standard.
    Missing/nonfinite values produce NaN rather than silently passing a test.
    """
    x = frame[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    failed = (x > OBJECTIVES).sum(axis=1)
    excursion = np.maximum(x / OBJECTIVES - 1, 0).sum(axis=1)
    f1 = 100 * failed / 12
    nse = excursion / 12
    f3 = nse / (0.01 * nse + 0.01)
    result = 100 - np.sqrt(2 * f1**2 + f3**2) / 1.732
    result[~np.isfinite(x).all(axis=1)] = np.nan
    return result


def quality_features(frame):
    """Mask invalid readings without deleting pollution extremes or guessing units.

    pH outside [0,14] and negative concentrations become missing. Temperature
    outside [-5,50] C is flagged for unit/measurement review, not converted.
    Every input gets a fixed missing/invalid flag, even if train has no missingness.
    """
    absent = set(FEATURES) - set(frame.columns)
    if absent:
        raise ValueError(f"Missing required measurement columns: {sorted(absent)}")
    x = frame[FEATURES].apply(pd.to_numeric, errors="coerce").astype(float)
    x = x.replace([np.inf, -np.inf], np.nan)
    for col in FEATURES:
        if col == "pH (ph units)":
            invalid = ~x[col].between(0, 14)
        elif col == "Temperature (cel)":
            invalid = ~x[col].between(-5, 50)
        else:
            invalid = x[col] < 0
        x.loc[invalid, col] = np.nan
    flags = x.isna().astype(float)
    flags.columns = [f"{col}__missing" for col in FEATURES]
    return pd.concat([x, flags], axis=1)


def read_dataset(path):
    df = pd.read_csv(path)
    required = set(META + FEATURES + [TARGET, "CCME_WQI"])
    if required - set(df.columns):
        raise ValueError(f"Dataset lacks columns: {sorted(required - set(df.columns))}")
    return df


def prepare(frame):
    """Keep source row IDs for traceability; never impute a target or a date."""
    df = frame.copy()
    df.insert(0, "source_row", np.arange(len(df)))
    duplicates = df.drop(columns="source_row").duplicated()
    date = pd.to_datetime(df.Date, format="%d-%m-%Y", errors="coerce")
    y = pd.to_numeric(df[TARGET], errors="coerce")
    invalid = date.isna() | ~y.between(0, 100) | df.Country.isna() | df.Area.isna()
    counts = {
        "exact_duplicates_removed": int(duplicates.sum()),
        "invalid_metadata_or_target": int(invalid.sum()),
    }
    df = df.loc[~duplicates & ~invalid].copy()
    df["parsed_date"] = date.loc[df.index]
    df[TARGET] = y.loc[df.index]
    return df, counts


def temporal_split(df, config):
    bounds = [pd.Timestamp(config[k]) for k in ["train_end", "validation_end", "calibration_end"]]
    if not bounds[0] < bounds[1] < bounds[2]:
        raise ValueError("Split boundaries must increase strictly")
    date = df.parsed_date
    masks = [
        date <= bounds[0],
        (date > bounds[0]) & (date <= bounds[1]),
        (date > bounds[1]) & (date <= bounds[2]),
        date > bounds[2],
    ]
    splits = dict(
        zip(["train", "validation", "calibration", "test"], [df.loc[mask].copy() for mask in masks])
    )
    if any(part.empty for part in splits.values()):
        raise ValueError("Every chronological partition must contain observations")
    return splits


def sample(frame, cap, seed):
    if cap and len(frame) > cap:
        return frame.sample(n=cap, random_state=seed).sort_index()
    return frame
