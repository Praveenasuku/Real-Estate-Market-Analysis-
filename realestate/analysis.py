"""Market-level analysis: summary, location premiums, affordability and segments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .data import PROXIMITY_ORDER


def market_summary(df: pd.DataFrame) -> dict:
    return {
        "districts": int(len(df)),
        "median_value": float(df["median_house_value"].median()),
        "median_income_usd": float(df["income_usd"].median()),
        "median_price_to_income": round(float(df["price_to_income"].median()), 2),
        "share_value_capped": round(float(df["value_capped"].mean()), 4),
    }


def proximity_premium(df: pd.DataFrame) -> pd.DataFrame:
    """Median value by ocean proximity, and the premium over inland districts.

    The raw premium mixes location with income, so we also compare districts
    within the same income band (quintile) to isolate the location effect.
    """
    table = (
        df.groupby("ocean_proximity")
        .agg(districts=("median_house_value", "size"), median_value=("median_house_value", "median"))
        .reindex([p for p in PROXIMITY_ORDER if p in df["ocean_proximity"].unique()])
    )
    inland = table.loc["INLAND", "median_value"]
    table["premium_vs_inland"] = (table["median_value"] / inland - 1).round(3)

    banded = df.assign(income_band=pd.qcut(df["median_income"], 5, labels=False))
    medians = banded.groupby(["income_band", "ocean_proximity"])["median_house_value"].median().unstack()
    within_band = (medians.div(medians["INLAND"], axis=0) - 1).median()
    table["premium_same_income"] = within_band.reindex(table.index).round(3)
    return table


def affordability_by_city(df: pd.DataFrame) -> pd.DataFrame:
    """Price-to-income ratio by nearest metro, most expensive first."""
    return (
        df.groupby("nearest_city")
        .agg(
            districts=("median_house_value", "size"),
            median_value=("median_house_value", "median"),
            median_income_usd=("income_usd", "median"),
            price_to_income=("price_to_income", "median"),
        )
        .sort_values("price_to_income", ascending=False)
        .round({"price_to_income": 2})
    )


SEGMENT_FEATURES = ["median_income", "median_house_value", "dist_to_city_km", "housing_median_age", "people_per_household"]


def segment_markets(df: pd.DataFrame, k: int = 4, random_state: int = 42) -> tuple[pd.Series, pd.DataFrame]:
    """K-Means market segments, named by their price and distance profile."""
    X = np.log1p(df[SEGMENT_FEATURES].clip(lower=0))
    labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(StandardScaler().fit_transform(X))

    profile = df.assign(segment=labels).groupby("segment").agg(
        districts=("median_house_value", "size"),
        median_value=("median_house_value", "median"),
        median_income_usd=("income_usd", "median"),
        median_dist_km=("dist_to_city_km", "median"),
        median_age=("housing_median_age", "median"),
    )
    names = _name_segments(profile)
    profile.index = profile.index.map(names)
    return pd.Series(labels, index=df.index).map(names), profile.sort_values("median_value", ascending=False)


def _name_segments(profile: pd.DataFrame) -> dict[int, str]:
    value_rank = profile["median_value"].rank()
    dist_rank = profile["median_dist_km"].rank()
    n = len(profile)
    names = {}
    for seg in profile.index:
        if value_rank[seg] == n:
            names[seg] = "Premium urban"
        elif value_rank[seg] == 1:
            names[seg] = "Affordable rural"
        elif dist_rank[seg] <= n / 2:
            names[seg] = "Mid-market urban"
        else:
            names[seg] = "Mid-market suburban"
    # Guarantee unique names if two segments share a profile.
    seen: dict[str, int] = {}
    for seg, name in names.items():
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            names[seg] = f"{name} {seen[name]}"
    return names
