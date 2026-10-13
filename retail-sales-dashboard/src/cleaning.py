"""
M1: Data Cleaning Module
========================
Objective: Prepare the Ecommerce Behavior Dataset for analysis.

This module handles:
- Dataset loading and understanding
- Missing value detection and handling
- Duplicate removal
- Data type conversions
- Saving cleaned dataset
"""

import pandas as pd
import numpy as np
import os


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CLEANED_FILE = os.path.join(PROCESSED_DIR, "cleaned_data.parquet")


# ---------------------------------------------------------------------------
# Sample data generator (used when Kaggle CSV is not present)
# ---------------------------------------------------------------------------
def generate_sample_data(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic Ecommerce Behavior Dataset so the
    dashboard works out-of-the-box without a Kaggle download.

    Column schema mirrors the real dataset:
        event_time, event_type, product_id, category_id, category_code,
        brand, price, user_id, user_session
    """
    rng = np.random.default_rng(seed)

    categories = {
        "electronics.smartphone": (150, 1200),
        "electronics.audio.headphone": (20, 400),
        "electronics.tablet": (100, 900),
        "computers.notebook": (300, 2000),
        "computers.accessories": (5, 200),
        "appliances.kitchen.blender": (30, 250),
        "appliances.washing_machine": (200, 1500),
        "kids.toys": (5, 100),
        "furniture.bedroom": (80, 1200),
        "sport.fitness": (10, 500),
    }
    brands = ["samsung", "apple", "xiaomi", "lg", "sony", "bosch", "ikea",
              "nike", "adidas", "lenovo", "hp", "asus", "philips", "canon"]

    cat_keys = list(categories.keys())
    chosen_cats = rng.choice(cat_keys, size=n)

    prices = np.array([
        rng.uniform(*categories[c]) for c in chosen_cats
    ]).round(2)

    # ~70 % view, ~20 % cart, ~10 % purchase
    event_types = rng.choice(
        ["view", "cart", "purchase"],
        size=n,
        p=[0.70, 0.20, 0.10],
    )

    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2020-11-30")
    seconds_range = int((end - start).total_seconds())
    random_seconds = rng.integers(0, seconds_range, size=n)
    event_times = [start + pd.Timedelta(seconds=int(s)) for s in random_seconds]

    user_ids = rng.integers(10_000_000, 99_999_999, size=n)
    product_ids = rng.integers(1_000_000, 9_999_999, size=n)
    category_ids = rng.integers(100_000_000, 999_999_999, size=n)
    session_ids = [f"sess_{rng.integers(100000, 999999)}" for _ in range(n)]
    chosen_brands = rng.choice(brands, size=n)

    # Inject ~2 % missing values in brand & category_code
    brand_arr = np.array(chosen_brands, dtype=object)
    cat_arr = np.array(chosen_cats, dtype=object)
    missing_mask = rng.random(n) < 0.02
    brand_arr[missing_mask] = np.nan
    cat_arr[rng.random(n) < 0.02] = np.nan

    df = pd.DataFrame({
        "event_time": event_times,
        "event_type": event_types,
        "product_id": product_ids,
        "category_id": category_ids,
        "category_code": cat_arr,
        "brand": brand_arr,
        "price": prices,
        "user_id": user_ids,
        "user_session": session_ids,
    })

    # Add ~0.5 % duplicate rows
    dup_idx = rng.choice(df.index, size=int(n * 0.005), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    return df


# ---------------------------------------------------------------------------
# Core cleaning functions
# ---------------------------------------------------------------------------
def load_raw_data() -> pd.DataFrame:
    """
    Load dataset from data/raw/.
    Looks for any CSV file; falls back to synthetic data generator.
    """
    csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    if csv_files:
        path = os.path.join(RAW_DIR, csv_files[0])
        print(f"[Cleaning] Loading CSV: {path}")
        # Large files: read in chunks, sample 500k rows for performance
        try:
            df = pd.read_csv(path, nrows=500_000)
        except Exception as e:
            print(f"[Cleaning] CSV read error: {e}. Using synthetic data.")
            df = generate_sample_data()
    else:
        print("[Cleaning] No CSV found – generating synthetic data.")
        df = generate_sample_data()
    return df


def understand_dataset(df: pd.DataFrame) -> dict:
    """Return a summary dict for display in the dashboard."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "sample": df.head(5),
        "describe": df.describe(include="all"),
    }


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Missing value strategy:
    - brand        → fill with 'unknown' (categorical, high missing)
    - category_code → fill with 'unknown'
    - price        → drop rows where price is missing (< 0.1 %)
    """
    df = df.copy()
    df["brand"] = df["brand"].fillna("unknown")
    df["category_code"] = df["category_code"].fillna("unknown")
    df = df.dropna(subset=["price"])
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"[Cleaning] Removed {before - after} duplicate rows.")
    return df


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert event_time to datetime and price to float."""
    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df


def remove_invalid_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with zero or negative prices for purchase events."""
    mask = (df["event_type"] == "purchase") & (df["price"] <= 0)
    df = df[~mask].copy()
    return df


def get_missing_summary(df: pd.DataFrame) -> pd.Series:
    """Return missing value counts per column."""
    return df.isnull().sum()


def clean_data() -> pd.DataFrame:
    """
    Full cleaning pipeline:
    1. Load raw data
    2. Convert dtypes
    3. Handle missing values
    4. Remove duplicates
    5. Remove invalid prices
    6. Save to processed/
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Check cache
    if os.path.exists(CLEANED_FILE):
        print("[Cleaning] Loading cached cleaned data.")
        return pd.read_parquet(CLEANED_FILE)

    df = load_raw_data()
    df = convert_dtypes(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = remove_invalid_prices(df)

    # Save
    df.to_parquet(CLEANED_FILE, index=False)
    print(f"[Cleaning] Saved cleaned data → {CLEANED_FILE}  shape={df.shape}")
    return df


# ---------------------------------------------------------------------------
# Entry point (run standalone for testing)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = clean_data()
    info = understand_dataset(df)
    print("Shape:", info["shape"])
    print("Columns:", info["columns"])
    print(info["sample"])
