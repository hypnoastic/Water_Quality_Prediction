"""Point scores and held-out residual intervals (no stationarity guarantee)."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y, prediction):
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)),
    }


def residual_quantile(y, prediction, alpha=0.1):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    residual = np.abs(np.asarray(y) - np.asarray(prediction))
    if not len(residual) or not np.isfinite(residual).all():
        raise ValueError("Calibration requires finite observations")
    rank = int(np.ceil((len(residual) + 1) * (1 - alpha)))
    if rank > len(residual):
        return float("inf")
    return float(np.partition(residual, rank - 1)[rank - 1])


def intervals(prediction, radius):
    return np.clip(prediction - radius, 0, 100), np.clip(prediction + radius, 0, 100)


def interval_metrics(y, prediction, radius):
    lower, upper = intervals(prediction, radius)
    return {
        "coverage": float(np.mean((y >= lower) & (y <= upper))),
        "mean_width": float(np.mean(upper - lower)),
        "radius": radius,
    }
