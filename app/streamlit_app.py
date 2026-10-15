"""Interactive dashboard for the California real estate market analysis.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from realestate.analysis import affordability_by_city, market_summary, proximity_premium, segment_markets  # noqa: E402
from realestate.data import PROXIMITY_ORDER, add_features, load_dataset  # noqa: E402
from realestate.model import MODEL_PATH, estimate_value, load_estimates, load_model, priced_below_model  # noqa: E402

st.set_page_config(page_title="Real Estate Market Analysis", page_icon="🏠", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    df = load_dataset()
    df["segment"], _ = segment_markets(df)
    return df


@st.cache_resource
def get_model():
    return load_model() if MODEL_PATH.exists() else None


df = get_data()
bundle = get_model()

st.title("California Real Estate Market Analysis")
st.caption("20,640 census districts (1990 California Housing dataset). Values are district medians in 1990 US dollars.")

with st.sidebar:
    st.header("Filters")
    cities = st.multiselect("Nearest metro", sorted(df["nearest_city"].unique()), default=sorted(df["nearest_city"].unique()))
    proximity = st.multiselect(
        "Ocean proximity",
        [p for p in PROXIMITY_ORDER if p in df["ocean_proximity"].unique()],
        default=[p for p in PROXIMITY_ORDER if p in df["ocean_proximity"].unique()],
    )
    income = st.slider("Median income ($K)", 0, 150, (0, 150), step=5)

view = df[
    df["nearest_city"].isin(cities)
    & df["ocean_proximity"].isin(proximity)
    & (df["income_usd"] >= income[0] * 1000)
    # The census caps income at $150,001, so the top of the slider means "no upper limit".
    & ((df["income_usd"] <= income[1] * 1000) | (income[1] >= 150))
]
if view.empty:
    st.warning("No districts match these filters. Widen the income range or add more areas.")
    st.stop()

overview, location, segments_tab, valuation, leads = st.tabs(
    ["Market overview", "Location and affordability", "Market segments", "Valuation tool", "Priced below model"]
)

with overview:
    s = market_summary(view)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Districts", f"{s['districts']:,}")
    c2.metric("Median home value", f"${s['median_value']:,.0f}")
    c3.metric("Median household income", f"${s['median_income_usd']:,.0f}")
    c4.metric("Price-to-income ratio", f"{s['median_price_to_income']:.2f}x")

    left, right = st.columns([3, 2])
    with left:
        fig = px.scatter(
            view, x="longitude", y="latitude", color="median_house_value", color_continuous_scale="Viridis",
            hover_data={"nearest_city": True, "median_house_value": ":$,.0f", "median_income": ":.2f"},
            labels={"median_house_value": "Value ($)"}, title="Median value by district", height=520,
        )
        fig.update_traces(marker={"size": 3})
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.histogram(view, x="median_house_value", nbins=50, title="Distribution of district values", height=260)
        fig.update_layout(xaxis_title="Median value ($)", yaxis_title="Districts")
        st.plotly_chart(fig, use_container_width=True)
        fig = px.scatter(
            view.sample(min(len(view), 4000), random_state=1), x="income_usd", y="median_house_value",
            opacity=0.35, title="Income drives value", height=260,
            labels={"income_usd": "Median income ($)", "median_house_value": "Median value ($)"},
        )
        st.plotly_chart(fig, use_container_width=True)
    if s["share_value_capped"] > 0:
        st.info(f"{s['share_value_capped']:.1%} of these districts sit at the census cap of $500,001, so their true value is higher.")

with location:
    prem = proximity_premium(df)
    st.subheader("What being near the ocean is worth")
    st.write("Comparing districts **in the same income band** separates the location effect from the fact that richer people live near the coast.")
    show = prem.drop(index=["ISLAND"], errors="ignore").reset_index()
    show["premium_same_income"] = (show["premium_same_income"] * 100).round(0)
    fig = px.bar(show[show["ocean_proximity"] != "INLAND"], x="premium_same_income", y="ocean_proximity", orientation="h",
                 text="premium_same_income", labels={"premium_same_income": "Premium over inland (%)", "ocean_proximity": ""},
                 height=280)
    fig.update_traces(texttemplate="+%{text:.0f}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Affordability by metro")
    afford = affordability_by_city(view).reset_index()
    afford.columns = ["Nearest metro", "Districts", "Median value ($)", "Median income ($)", "Price-to-income"]
    st.dataframe(afford, hide_index=True, use_container_width=True)

with segments_tab:
    st.subheader("Four market segments (K-Means on price, income, distance, age and density)")
    profile = view.groupby("segment").agg(
        districts=("median_house_value", "size"),
        median_value=("median_house_value", "median"),
        median_income=("income_usd", "median"),
        median_distance_km=("dist_to_city_km", "median"),
        median_age_years=("housing_median_age", "median"),
    ).sort_values("median_value", ascending=False).round(0)
    st.dataframe(profile, use_container_width=True)
    fig = px.scatter(view, x="longitude", y="latitude", color="segment", height=520, title="Segments on the map")
    fig.update_traces(marker={"size": 3})
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)

with valuation:
    st.subheader("Estimate a district's median value")
    if bundle is None:
        st.error("Model file not found. Run: python -m realestate.report")
    else:
        c1, c2, c3 = st.columns(3)
        lat = c1.number_input("Latitude", 32.5, 42.0, 34.05, step=0.01)
        lon = c1.number_input("Longitude", -124.5, -114.0, -118.24, step=0.01)
        prox = c1.selectbox("Ocean proximity", ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY"])
        inc = c2.number_input("Median income ($K)", 5.0, 150.0, 45.0, step=1.0)
        age = c2.slider("Median house age (years)", 1, 52, 25)
        rooms = c2.number_input("Rooms per household", 1.0, 15.0, 5.0, step=0.1)
        beds = c3.number_input("Bedrooms per room", 0.1, 0.6, 0.2, step=0.01)
        people = c3.number_input("People per household", 1.0, 8.0, 2.8, step=0.1)
        households = c3.number_input("Households", 50, 5000, 450, step=10)

        row = pd.DataFrame([{
            "longitude": lon, "latitude": lat, "housing_median_age": age, "median_income": inc / 10,
            "total_rooms": rooms * households, "total_bedrooms": beds * rooms * households,
            "population": people * households, "households": households, "ocean_proximity": prox,
            "median_house_value": np.nan,
        }])
        row = add_features(row)
        estimate = float(estimate_value(bundle, row)[0])
        st.metric("Estimated median value", f"${estimate:,.0f}")
        st.caption(f"Nearest metro: {row['nearest_city'][0]} ({row['dist_to_city_km'][0]:.0f} km). "
                   "Typical error on unseen areas is about $39K, so treat this as a range, not a price.")

with leads:
    st.subheader("Districts priced well below the model's estimate")
    st.write("Each district is valued by a model that never saw it (out-of-fold). These are **leads for research, "
             "not buy signals**: the gap can come from things the data doesn't capture, like housing condition.")
    ratio = st.slider("Actual value / model estimate below", 0.4, 0.9, 0.7, step=0.05)
    table = priced_below_model(view, load_estimates(), ratio=ratio)
    st.write(f"**{len(table):,} districts** match.")
    table = table.assign(
        median_income=(table["median_income"] * 10_000).round(0),
        actual_vs_model=(table["actual_vs_model"] * 100).round(0),
    ).rename(columns={
        "nearest_city": "Nearest metro", "ocean_proximity": "Proximity", "latitude": "Lat", "longitude": "Lon",
        "median_income": "Income ($)", "median_house_value": "Actual ($)", "model_estimate": "Estimate ($)",
        "actual_vs_model": "Actual vs estimate (%)",
    })
    st.dataframe(table.head(200), hide_index=True, use_container_width=True)
