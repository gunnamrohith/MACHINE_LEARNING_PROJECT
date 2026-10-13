"""
M5: Regression Analysis Module
================================
Objective: Build a simple predictive model to predict product price.

Pipeline:
1. Feature selection from purchase events
2. Categorical encoding (category_main, brand)
3. Category-level mean price as a numeric feature (improves R²)
4. 80/20 train-test split
5. LinearRegression from scikit-learn
6. Evaluation: MAE, RMSE, R²
7. Prediction interface for the dashboard
"""

import pandas as pd
import numpy as np
import os
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "model.pkl")


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------
def prepare_features(df: pd.DataFrame):
    """
    Select and encode features for the regression model.

    Features (X):
        hour, weekday,
        category_main_enc  (label-encoded),
        brand_enc          (label-encoded),
        cat_mean_price     (average price per category – strong numeric signal)

    Target (y):
        price
    """
    purchases = df[df["event_type"] == "purchase"].copy()
    purchases = purchases.dropna(subset=["price", "category_main", "brand", "hour", "weekday"])

    # Category-level mean price — gives the model a strong numeric signal
    cat_mean = purchases.groupby("category_main")["price"].transform("mean")
    purchases["cat_mean_price"] = cat_mean

    # Encode categoricals
    encoders = {}
    for col in ["category_main", "brand"]:
        le = LabelEncoder()
        purchases[f"{col}_enc"] = le.fit_transform(purchases[col].astype(str))
        encoders[col] = le

    # Store category → mean price mapping for inference
    cat_mean_map = purchases.groupby("category_main")["price"].mean().to_dict()
    global_mean = purchases["price"].mean()

    feature_cols = ["hour", "weekday", "category_main_enc", "brand_enc", "cat_mean_price"]
    X = purchases[feature_cols].values
    y = purchases["price"].values

    return X, y, encoders, feature_cols, purchases, cat_mean_map, global_mean


def train_model(df: pd.DataFrame) -> dict:
    """Train a Linear Regression model and return evaluation results."""
    X, y, encoders, feature_cols, purchases, cat_mean_map, global_mean = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    artifacts = {
        "model": model,
        "encoders": encoders,
        "feature_cols": feature_cols,
        "cat_mean_map": cat_mean_map,
        "global_mean": global_mean,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "categories": list(encoders["category_main"].classes_),
        "brands": list(encoders["brand"].classes_),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    save_pkg = {k: v for k, v in artifacts.items()
                if k not in ("X_test", "y_test", "y_pred")}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(save_pkg, f)

    return artifacts


def load_or_train(df: pd.DataFrame) -> dict:
    """Load cached model or train a new one, always returning test metrics."""
    # Always retrain to get fresh y_test / y_pred arrays (needed for plots)
    return train_model(df)


def predict_price(model_artifacts: dict, category: str, brand: str,
                  hour: int, weekday: int) -> float:
    """
    Predict product price for given inputs.

    Parameters
    ----------
    model_artifacts : dict  returned by train_model
    category        : str   top-level category (e.g. 'electronics')
    brand           : str   brand name (e.g. 'samsung')
    hour            : int   0–23
    weekday         : int   0=Monday … 6=Sunday
    """
    encoders     = model_artifacts["encoders"]
    model        = model_artifacts["model"]
    cat_mean_map = model_artifacts["cat_mean_map"]
    global_mean  = model_artifacts["global_mean"]

    def safe_encode(le, val):
        classes = list(le.classes_)
        return classes.index(val) if val in classes else 0

    cat_enc   = safe_encode(encoders["category_main"], category)
    brand_enc = safe_encode(encoders["brand"], brand)
    cat_mean  = cat_mean_map.get(category, global_mean)

    features = np.array([[hour, weekday, cat_enc, brand_enc, cat_mean]])
    predicted = model.predict(features)[0]
    return max(0.0, float(predicted))


def model_interpretation() -> dict:
    return {
        "strengths": [
            "Simple and highly interpretable — easy to understand for beginners.",
            "Fast to train, even on large datasets with 100k+ rows.",
            "Provides a solid baseline to compare against advanced models.",
            "Category mean price feature captures most real-world price variation.",
        ],
        "limitations": [
            "Assumes a linear relationship between features and price.",
            "Only 5 features used; adding more (popularity score, day-of-month) would help.",
            "Cannot capture non-linear interactions between brand and category.",
            "Outlier products (luxury items) can inflate RMSE significantly.",
        ],
        "beginner_note": (
            "**R²** tells us what fraction of price variation is explained by the model. "
            "An R² of 0.60 means the model explains 60% of why prices differ — "
            "a decent score for a 5-feature linear model! "
            "**MAE** is the average rupee difference between predicted and actual price. "
            "**RMSE** penalises large errors more than MAE, so it is higher when outliers exist."
        ),
    }
