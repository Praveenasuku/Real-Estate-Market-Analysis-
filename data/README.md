# Data

`housing.csv` is the **California Housing** dataset: one row per census block group ("district") from the 1990 US Census, 20,640 rows.

- Original source: R. Kelley Pace and Ronald Barry, "Sparse Spatial Autoregressions", Statistics and Probability Letters, 1997 (StatLib).
- This copy, with the `ocean_proximity` column, comes from Aurélien Géron's *Hands-On Machine Learning* repository: https://github.com/ageron/handson-ml2

Known quirks handled in `realestate/data.py`:

- 207 districts have no `total_bedrooms` value.
- `median_house_value` is capped at $500,001 and floored at $14,999, and `housing_median_age` is capped at 52.
- `median_income` is in tens of thousands of dollars.
