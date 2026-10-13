"""
M2 (Part 1): Feature Engineering Module
========================================
Objective: Create useful time-based and business features for analysis.

Features created:
- year, month, day, hour      → time decomposition
- weekday                     → Mon=0 … Sun=6
- is_weekend                  → bool flag
- revenue                     → alias for price (on purchase events)
- category_main               → top-level category (before first '.')
"""

import pandas as pd
import numpy as np
import os

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FEATURED_FILE = os.path.join(PROCESSED_DIR, "featured_data.parquet")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose event_time into year, month, day, hour, weekday."""
    df = df.copy()
    dt = df["event_time"].dt
    df["year"] = dt.year
    df["month"] = dt.month
    df["day"] = dt.day
    df["hour"] = dt.hour
    df["weekday"] = dt.dayofweek          # 0=Monday, 6=Sunday
    df["weekday_name"] = dt.day_name()
    df["is_weekend"] = df["weekday"].isin([5, 6])
    df["month_name"] = dt.strftime("%b")
    df["date"] = dt.date
    return df


def add_revenue_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Revenue = Price, but only for 'purchase' events.
    Non-purchase rows get revenue = 0.
    """
    df = df.copy()
    df["revenue"] = np.where(df["event_type"] == "purchase", df["price"], 0.0)
    return df


def add_category_main(df: pd.DataFrame) -> pd.DataFrame:
    """Extract top-level category from dot-separated category_code."""
    df = df.copy()
    df["category_main"] = df["category_code"].str.split(".").str[0]
    df["category_main"] = df["category_main"].fillna("unknown")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = add_time_features(df)
    df = add_revenue_feature(df)
    df = add_category_main(df)
    return df


def load_or_build(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Load cached featured data or build from cleaned DataFrame."""
    if os.path.exists(FEATURED_FILE):
        return pd.read_parquet(FEATURED_FILE)
    df = engineer_features(cleaned_df)
    df.to_parquet(FEATURED_FILE, index=False)
    print(f"[Features] Saved featured data → {FEATURED_FILE}  shape={df.shape}")
    return df


if __name__ == "__main__":
    from cleaning import clean_data
    cleaned = clean_data()
    featured = load_or_build(cleaned)
    print(featured[["event_time", "year", "month", "hour", "weekday",
                     "is_weekend", "revenue", "category_main"]].head())
