"""Train the model and produce the figures and summary used in the README.

Usage:
    python -m realestate.report
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analysis import affordability_by_city, market_summary, proximity_premium, segment_markets  # noqa: E402
from .data import ROOT, load_dataset  # noqa: E402
from .model import load_estimates, priced_below_model, train  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures"


def main() -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "font.size": 10})

    df = load_dataset()
    metrics = train(df)
    estimates = load_estimates()
    premium = proximity_premium(df)
    afford = affordability_by_city(df)
    _, segments = segment_markets(df)
    leads = priced_below_model(df, estimates)

    # 1. Price map
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    sc = ax.scatter(df["longitude"], df["latitude"], c=np.log10(df["median_house_value"]), s=2, cmap="viridis")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    ticks = [5e4, 1e5, 2e5, 5e5]
    cbar.set_ticks(np.log10(ticks))
    cbar.set_ticklabels(["$50K", "$100K", "$200K", "$500K"])
    ax.set_title("Median house value by district")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "price_map.png")
    plt.close(fig)

    # 2. Location premium at the same income level
    prem = premium.drop(index=["INLAND", "ISLAND"], errors="ignore")["premium_same_income"] * 100
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.barh(prem.index, prem.values, color="#1f6feb")
    for i, v in enumerate(prem.values):
        ax.text(v + 1, i, f"+{v:.0f}%", va="center")
    ax.set_xlabel("Value premium over inland districts, same income band (%)")
    ax.set_xlim(0, prem.max() * 1.25)
    ax.set_title("What location is worth")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "location_premium.png")
    plt.close(fig)

    # 3. Model fit on districts it never saw
    data = df.join(estimates, how="inner")
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(data["model_estimate"] / 1000, data["median_house_value"] / 1000, s=2, alpha=0.3, color="#2da44e")
    lim = [0, 520]
    ax.plot(lim, lim, color="#d1242f", lw=1, label="perfect estimate")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Model estimate ($K, out-of-fold)")
    ax.set_ylabel("Actual median value ($K)")
    ax.set_title("Valuation model accuracy")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_fit.png")
    plt.close(fig)

    summary = {
        "market": market_summary(df),
        "premium_same_income": premium["premium_same_income"].drop(index=["ISLAND"], errors="ignore").to_dict(),
        "affordability": afford[["median_value", "median_income_usd", "price_to_income"]].to_dict(orient="index"),
        "segments": segments.round(1).to_dict(orient="index"),
        "model": metrics,
        "priced_below_model_count": int(len(leads)),
    }
    (ROOT / "reports" / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    return summary


if __name__ == "__main__":
    main()
