"""Métricas de avaliação para regressão (estatísticas + métricas de negócio)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def regression_metrics(y_true, y_pred, n_features: int | None = None) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pct_err = np.abs(y_pred - y_true) / np.abs(y_true)

    out = {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2,
        "median_ae": median_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "bias": float(np.mean(y_pred - y_true)),          # >0 = superestima em média
        "within_10pct": float(np.mean(pct_err <= 0.10) * 100),
        "within_20pct": float(np.mean(pct_err <= 0.20) * 100),
    }
    if n_features is not None and n - n_features - 1 > 0:
        out["adj_r2"] = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return out
