import json

import numpy as np
import pandas as pd
import pytest

from realestate.analysis import affordability_by_city, proximity_premium, segment_markets
from realestate.data import ROOT, add_features, haversine_km, load_dataset
from realestate.model import load_estimates, priced_below_model


@pytest.fixture(scope="module")
def df():
    return load_dataset()


def test_haversine_known_distance():
    # San Francisco to Los Angeles is about 559 km.
    assert haversine_km(37.7749, -122.4194, 34.0522, -118.2437) == pytest.approx(559, abs=5)


def test_cleaning_fills_missing_bedrooms_and_flags_caps(df):
    assert df["total_bedrooms"].isna().sum() == 0
    assert df["value_capped"].sum() == (df["median_house_value"] >= 500_001).sum()
    assert len(df) == 20_640


def test_features_are_sensible(df):
    assert (df["rooms_per_household"] > 0).all()
    assert df["bedrooms_ratio"].between(0, 1.5).all()
    assert set(df["nearest_city"]) <= {"San Francisco", "Los Angeles", "San Diego", "Sacramento", "San Jose"}


def test_add_features_on_single_row():
    row = pd.DataFrame([{
        "longitude": -118.24, "latitude": 34.05, "housing_median_age": 20, "median_income": 5.0,
        "total_rooms": 1000, "total_bedrooms": 200, "population": 600, "households": 200,
        "ocean_proximity": "<1H OCEAN", "median_house_value": 300_000,
    }])
    out = add_features(row)
    assert out.loc[0, "nearest_city"] == "Los Angeles"
    assert out.loc[0, "price_to_income"] == pytest.approx(6.0)


def test_coastal_premium_is_positive_at_same_income(df):
    prem = proximity_premium(df)
    for area in ["<1H OCEAN", "NEAR OCEAN", "NEAR BAY"]:
        assert prem.loc[area, "premium_same_income"] > 0.2


def test_affordability_sorted_most_expensive_first(df):
    ratios = affordability_by_city(df)["price_to_income"].to_numpy()
    assert np.all(ratios[:-1] >= ratios[1:])


def test_segments_have_unique_names(df):
    labels, profile = segment_markets(df)
    assert labels.notna().all()
    assert profile.index.is_unique
    assert len(profile) == 4


def test_model_metrics_are_reasonable():
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    assert metrics["random_split_test"]["r2"] > 0.75
    # Unseen areas are harder, and the spatial score should show that.
    assert metrics["spatial_holdout_5fold"]["mae_usd"] > metrics["random_split_test"]["mae_usd"]


def test_priced_below_model_respects_ratio(df):
    leads = priced_below_model(df, load_estimates(), ratio=0.7)
    assert (leads["actual_vs_model"] < 0.7).all()
    assert (leads["median_house_value"] > 14_999).all()
