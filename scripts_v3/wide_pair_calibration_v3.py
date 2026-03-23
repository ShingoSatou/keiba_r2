from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def fit_wide_pair_calibrator(
    raw_scores: np.ndarray,
    labels: np.ndarray,
    *,
    method: str = "isotonic",
) -> dict[str, Any]:
    scores = np.asarray(raw_scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    valid = np.isfinite(scores) & np.isfinite(y)
    if not valid.any():
        raise ValueError("No finite samples available for wide pair calibration.")

    scores = scores[valid]
    y = y[valid]
    if method not in {"isotonic", "logreg"}:
        raise ValueError(f"Unsupported wide pair calibration method: {method}")

    if np.unique(scores).size <= 1 or np.unique(y).size <= 1:
        model: Any = {"constant_prob": float(np.mean(y))}
    elif method == "logreg":
        model = LogisticRegression(
            random_state=42,
            max_iter=2000,
            solver="lbfgs",
        )
        model.fit(scores.reshape(-1, 1), y.astype(int))
    else:
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(scores, y)

    return {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "method": method,
        "input_col": "p_wide_raw",
        "output_col": "p_wide",
        "model": model,
    }


def predict_wide_pair_calibrator(bundle: dict[str, Any], raw_scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(raw_scores, dtype=float)
    method = str(bundle.get("method", ""))
    model = bundle.get("model")
    if method not in {"isotonic", "logreg"}:
        raise ValueError(f"Unsupported wide pair calibration method: {method}")

    valid = np.isfinite(scores)
    pred = np.full(scores.shape, np.nan, dtype=float)
    if not valid.any():
        return pred

    if isinstance(model, dict):
        pred[valid] = float(model.get("constant_prob", 0.0))
        return np.clip(pred, 0.0, 1.0)

    if method == "logreg":
        pred_valid = model.predict_proba(scores[valid].reshape(-1, 1))[:, 1]
    else:
        pred_valid = model.predict(scores[valid])
    pred[valid] = np.asarray(pred_valid, dtype=float)
    return np.clip(pred, 0.0, 1.0)


def apply_wide_pair_calibrator(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
    *,
    raw_col: str = "p_wide_raw",
    output_col: str = "p_wide",
) -> pd.DataFrame:
    if raw_col not in frame.columns:
        raise ValueError(f"Missing raw score column for wide pair calibration: {raw_col}")
    out = frame.copy()
    out[raw_col] = pd.to_numeric(out[raw_col], errors="coerce")
    out[output_col] = predict_wide_pair_calibrator(
        bundle,
        out[raw_col].to_numpy(dtype=float),
    )
    return out
