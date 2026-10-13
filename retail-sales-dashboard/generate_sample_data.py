"""
generate_sample_data.py
========================
Standalone script to generate a richer 50,000-row synthetic dataset
and save it to data/raw/sample_ecommerce.csv

Run: python3 generate_sample_data.py

The generated data has REALISTIC price correlations per category, making
the regression model more meaningful.
"""

import pandas as pd
import numpy as np
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "raw", "sample_ecommerce.csv")
N = 50_000
SEED = 42


def main():
    rng = np.random.default_rng(SEED)

    # Category definitions: (base_price, std, sub_categories)
    category_config = {
        "electronics": {
            "base": 600, "std": 400,
            "subs": ["electronics.smartphone", "electronics.audio.headphone",
                     "electronics.tablet", "electronics.camera"],
            "brands": ["samsung", "apple", "xiaomi", "sony", "lg"],
            "weight": 0.20,
        },
        "computers": {
            "base": 800, "std": 500,
            "subs": ["computers.notebook", "computers.desktop", "computers.accessories"],
            "brands": ["lenovo", "hp", "asus", "dell", "apple"],
            "weight": 0.15,
        },
        "appliances": {
            "base": 500, "std": 350,
            "subs": ["appliances.kitchen.blender", "appliances.washing_machine",
                     "appliances.refrigerator"],
            "brands": ["bosch", "lg", "samsung", "whirlpool", "philips"],
            "weight": 0.15,
        },
        "furniture": {
            "base": 300, "std": 250,
            "subs": ["furniture.bedroom", "furniture.living_room", "furniture.office"],
            "brands": ["ikea", "nilkamal", "hometown", "godrej", "urban_ladder"],
            "weight": 0.10,
        },
        "sport": {
            "base": 120, "std": 100,
            "subs": ["sport.fitness", "sport.outdoor", "sport.team"],
            "brands": ["nike", "adidas", "puma", "reebok", "decathlon"],
            "weight": 0.10,
        },
        "kids": {
            "base": 60, "std": 50,
            "subs": ["kids.toys", "kids.clothing", "kids.books"],
            "brands": ["mattel", "hasbro", "fisher_price", "lego", "vtech"],
            "weight": 0.10,
        },
        "fashion": {
            "base": 80, "std": 70,
            "subs": ["fashion.clothing", "fashion.shoes", "fashion.accessories"],
            "brands": ["zara", "h&m", "nike", "adidas", "levis"],
            "weight": 0.10,
        },
        "beauty": {
            "base": 50, "std": 40,
            "subs": ["beauty.skincare", "beauty.haircare", "beauty.makeup"],
            "brands": ["loreal", "lakme", "maybelline", "himalaya", "dove"],
            "weight": 0.10,
        },
    }

    cats = list(category_config.keys())
    weights = [category_config[c]["weight"] for c in cats]

    # Sample category for each row
    chosen_main_cats = rng.choice(cats, size=N, p=weights)

    rows = []
    start = pd.Timestamp("2020-01-01")
    end   = pd.Timestamp("2020-11-30")
    seconds_range = int((end - start).total_seconds())

    # Event type probabilities
    event_probs = [0.68, 0.21, 0.11]   # view, cart, purchase

    for i, main_cat in enumerate(chosen_main_cats):
        cfg = category_config[main_cat]

        # Pick sub-category and brand
        sub_cat = rng.choice(cfg["subs"])
        brand   = rng.choice(cfg["brands"])

        # Price: correlated with category, with some random noise
        price = max(1.0, rng.normal(cfg["base"], cfg["std"]))
        price = round(price, 2)

        # Event type
        event_type = rng.choice(["view", "cart", "purchase"], p=event_probs)

        # Timestamp (weighted toward business hours)
        rand_sec  = rng.integers(0, seconds_range)
        timestamp = start + pd.Timedelta(seconds=int(rand_sec))

        user_id    = rng.integers(10_000_000, 99_999_999)
        product_id = rng.integers(1_000_000,  9_999_999)
        cat_id     = rng.integers(100_000_000, 999_999_999)
        session_id = f"sess_{rng.integers(100_000, 999_999)}"

        rows.append({
            "event_time":    timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "event_type":    event_type,
            "product_id":    product_id,
            "category_id":   cat_id,
            "category_code": sub_cat if rng.random() > 0.02 else None,
            "brand":         brand    if rng.random() > 0.02 else None,
            "price":         price,
            "user_id":       user_id,
            "user_session":  session_id,
        })

    df = pd.DataFrame(rows)

    # Inject ~0.5% duplicates
    dup_idx = rng.choice(df.index, size=int(N * 0.005), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df):,} rows → {OUTPUT_PATH}")
    print(f"Columns: {list(df.columns)}")
    print(df.groupby("event_type").size())


if __name__ == "__main__":
    main()
