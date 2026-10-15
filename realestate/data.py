"""Load, clean and enrich the California Housing dataset (1990 census, 20,640 districts)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "housing.csv"

# The census capped these values, so the true numbers above them are unknown.
VALUE_CAP = 500_001
AGE_CAP = 52

# Major metro centres used to measure how far each district is from a city.
CITIES = {
    "San Francisco": (37.7749, -122.4194),
    "Los Angeles": (34.0522, -118.2437),
    "San Diego": (32.7157, -117.1611),
    "Sacramento": (38.5816, -121.4944),
    "San Jose": (37.3382, -121.8863),
}

PROXIMITY_ORDER = ["INLAND", "<1H OCEAN", "NEAR OCEAN", "NEAR BAY", "ISLAND"]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Works on scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-household ratios, affordability and distance to the nearest metro."""
    out = df.copy()
    out["rooms_per_household"] = out["total_rooms"] / out["households"]
    out["bedrooms_ratio"] = out["total_bedrooms"] / out["total_rooms"]
    out["people_per_household"] = out["population"] / out["households"]
    out["income_usd"] = out["median_income"] * 10_000
    out["price_to_income"] = out["median_house_value"] / out["income_usd"]

    distances = np.column_stack(
        [haversine_km(out["latitude"], out["longitude"], lat, lon) for lat, lon in CITIES.values()]
    )
    out["nearest_city"] = np.array(list(CITIES))[distances.argmin(axis=1)]
    out["dist_to_city_km"] = distances.min(axis=1)
    return out


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Clean the raw CSV and add features.

    - 207 districts have no bedroom count: fill with the median bedrooms ratio
      for the same ocean-proximity group, times the district's total rooms.
    - Flag districts at the census caps so analysis and modelling can treat
      them separately.
    """
    df = pd.read_csv(path)
    ratio = df["total_bedrooms"] / df["total_rooms"]
    group_ratio = ratio.groupby(df["ocean_proximity"]).transform("median")
    df["total_bedrooms"] = df["total_bedrooms"].fillna((group_ratio * df["total_rooms"]).round())

    df["value_capped"] = df["median_house_value"] >= VALUE_CAP
    df["age_capped"] = df["housing_median_age"] >= AGE_CAP
    return add_features(df)
