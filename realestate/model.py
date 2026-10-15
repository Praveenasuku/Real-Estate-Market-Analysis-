"""District valuation model and 'priced below the model' opportunity finder.

Two honest-evaluation choices:
- Districts at the $500,001 census cap are left out of training and scoring,
  because their true value is unknown.
- Besides a random split, we test on whole geographic areas the model never saw
  (grid cells), since neighbouring districts are similar and a random split
  flatters the score.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import PROXIMITY_ORDER, ROOT

MODEL_PATH = ROOT / "models" / "valuation_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
ESTIMATES_PATH = ROOT / "models" / "out_of_fold_estimates.csv"

NUMERIC = [
    "longitude", "latitude", "housing_median_age", "median_income", "rooms_per_household",
    "bedrooms_ratio", "people_per_household", "population", "dist_to_city_km",
]
CATEGORICAL = ["ocean_proximity"]
FEATURES = NUMERIC + CATEGORICAL


def build_model() -> Pipeline:
    prep = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC),
            ("cat", OneHotEncoder(categories=[PROXIMITY_ORDER], handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    reg = HistGradientBoostingRegressor(learning_rate=0.06, max_iter=600, l2_regularization=1.0, random_state=42)
    return Pipeline([("prep", prep), ("reg", reg)])


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae_usd": round(float(mean_absolute_error(y_true, y_pred))),
        "mape": round(float(mean_absolute_percentage_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def train(df: pd.DataFrame) -> dict:
    data = df[~df["value_capped"]].copy()
    X, y = data[FEATURES], data["median_house_value"].to_numpy()
    log_y = np.log(y)

    X_tr, X_te, y_tr, y_te = train_test_split(X, log_y, test_size=0.2, random_state=42)
    model = build_model().fit(X_tr, y_tr)
    random_split = _scores(np.exp(y_te), np.exp(model.predict(X_te)))

    # Spatial check: hold out whole 0.5-degree grid cells.
    cells = (np.floor(data["latitude"] / 0.5).astype(int).astype(str) + "_"
             + np.floor(data["longitude"] / 0.5).astype(int).astype(str))
    preds = np.zeros(len(data))
    for tr_idx, te_idx in GroupKFold(n_splits=5).split(X, log_y, groups=cells):
        fold = build_model().fit(X.iloc[tr_idx], log_y[tr_idx])
        preds[te_idx] = np.exp(fold.predict(X.iloc[te_idx]))
    spatial_split = _scores(y, preds)

    # Out-of-fold estimates: every district is valued by a model that never saw it.
    oof = np.exp(cross_val_predict(build_model(), X, log_y, cv=KFold(5, shuffle=True, random_state=42)))
    pd.DataFrame({"model_estimate": oof.round()}, index=data.index).to_csv(ESTIMATES_PATH, index_label="district_id")

    final = build_model().fit(X, log_y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "features": FEATURES}, MODEL_PATH)

    metrics = {
        "rows_used": int(len(data)),
        "rows_excluded_capped": int(df["value_capped"].sum()),
        "random_split_test": random_split,
        "spatial_holdout_5fold": spatial_split,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


def load_model(path: Path = MODEL_PATH) -> dict:
    return joblib.load(path)


def estimate_value(bundle: dict, rows: pd.DataFrame) -> np.ndarray:
    return np.exp(bundle["pipeline"].predict(rows[bundle["features"]]))


def load_estimates(path: Path = ESTIMATES_PATH) -> pd.Series:
    return pd.read_csv(path, index_col="district_id")["model_estimate"]


def priced_below_model(df: pd.DataFrame, estimates: pd.Series, ratio: float = 0.7, min_households: int = 100) -> pd.DataFrame:
    """Districts whose actual median value is well below the out-of-fold estimate.

    These are leads for further research, not buy signals: the gap can reflect
    things the data doesn't capture, such as housing condition or crime.
    """
    data = df.join(estimates, how="inner")
    # Skip the census floor ($14,999) too: like the cap, its true value is unknown.
    data = data[(data["households"] >= min_households) & (data["median_house_value"] > 14_999)].copy()
    data["actual_vs_model"] = data["median_house_value"] / data["model_estimate"]
    cols = ["nearest_city", "ocean_proximity", "latitude", "longitude", "median_income",
            "median_house_value", "model_estimate", "actual_vs_model"]
    return data.loc[data["actual_vs_model"] < ratio, cols].sort_values("actual_vs_model")
