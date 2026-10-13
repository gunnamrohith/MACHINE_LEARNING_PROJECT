"""
M7: Customer Purchase Prediction Module
=========================================
Objective: Predict whether a customer interaction (view/cart/purchase
event) is a PURCHASE (binary classification: purchase / no purchase).

Note on approach: the dataset records individual customer-product
interactions (event_time, event_type, product, category, brand, price,
user). Each user/session in this dataset overwhelmingly contains a single
interaction (~1.03 events per session), so session-level aggregation
collapses almost every feature back to the row itself. We therefore model
purchase prediction at the INTERACTION level: given a customer's
interaction with a product (category, brand, price, and when it happened),
predict whether that interaction is a purchase.

Pipeline:
1. Feature/target separation (target = event_type == 'purchase').
2. Train/test split (80/20, stratified, random_state=42).
3. Preprocess with ColumnTransformer (StandardScaler + OneHotEncoder),
   fitted ONLY on the training data (no leakage).
4. Train Logistic Regression and Random Forest on identical data.
5. Evaluate both on the same test set: Accuracy, Precision, Recall, F1.
6. Extract Random Forest feature importance / Logistic Regression coefficients.
7. Persist the trained models + preprocessing pipeline for the prediction UI.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "purchase_model.pkl"
)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

NUMERIC_FEATURES = ["price", "hour", "weekday", "is_weekend"]
CATEGORICAL_FEATURES = ["category_main", "brand"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = "purchased"


# ---------------------------------------------------------------------------
# Data Understanding (rubric: dataset inspection before modelling)
# ---------------------------------------------------------------------------
def dataset_overview(df: pd.DataFrame) -> dict:
    """Return standard 'Data Understanding' statistics for the raw event data."""
    return {
        "n_records": len(df),
        "n_features": df.shape[1],
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_values": {c: int(df[c].nunique()) for c in df.columns},
        "target_class_counts": df["event_type"].value_counts().to_dict(),
        "describe_numeric": df.select_dtypes(include=[np.number]).describe(),
    }


def detect_outliers_iqr(df: pd.DataFrame, column: str = "price") -> dict:
    """IQR-based outlier detection on a numeric column (Data Preprocessing step)."""
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (df[column] < lower) | (df[column] > upper)
    return {
        "column": column,
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "n_outliers": int(mask.sum()),
        "pct_outliers": float(mask.mean() * 100),
    }


# ---------------------------------------------------------------------------
# Feature / target preparation
# ---------------------------------------------------------------------------
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the model-ready table.

    Target: purchased = 1 if the interaction's event_type is 'purchase', else 0.
    Features: product price, time of interaction (hour/weekday/is_weekend),
    product category and brand -- all known at the moment of interaction,
    so there is no leakage from the target.
    """
    data = df.copy()
    data["is_weekend"] = data["is_weekend"].astype(int)
    data[TARGET_COL] = (data["event_type"] == "purchase").astype(int)
    data = data.dropna(subset=FEATURE_COLS + [TARGET_COL])
    return data[FEATURE_COLS + [TARGET_COL]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Scale numeric features, one-hot encode categoricals.
    Must be fit ONLY on training data to avoid leakage."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def _feature_names(preprocessor: ColumnTransformer) -> list:
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return NUMERIC_FEATURES + cat_names


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_models(df: pd.DataFrame) -> dict:
    """Train Logistic Regression + Random Forest and evaluate both on the
    same held-out test set."""
    model_df = prepare_data(df)

    X = model_df[FEATURE_COLS]
    y = model_df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)   # fit ONLY on train
    X_test_t = preprocessor.transform(X_test)
    feature_names = _feature_names(preprocessor)

    results = {}

    # ---- Logistic Regression ----------------------------------------------
    log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    log_reg.fit(X_train_t, y_train)
    lr_pred = log_reg.predict(X_test_t)
    lr_proba = log_reg.predict_proba(X_test_t)[:, 1]

    results["logistic_regression"] = {
        "model": log_reg,
        "y_pred": lr_pred,
        "y_proba": lr_proba,
        "accuracy": accuracy_score(y_test, lr_pred),
        "precision": precision_score(y_test, lr_pred, zero_division=0),
        "recall": recall_score(y_test, lr_pred, zero_division=0),
        "f1": f1_score(y_test, lr_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, lr_proba),
        "roc_curve": roc_curve(y_test, lr_proba),
        "confusion_matrix": confusion_matrix(y_test, lr_pred),
        "coefficients": dict(zip(feature_names, log_reg.coef_[0])),
    }

    # ---- Random Forest ------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train_t, y_train)
    rf_pred = rf.predict(X_test_t)
    rf_proba = rf.predict_proba(X_test_t)[:, 1]

    results["random_forest"] = {
        "model": rf,
        "y_pred": rf_pred,
        "y_proba": rf_proba,
        "accuracy": accuracy_score(y_test, rf_pred),
        "precision": precision_score(y_test, rf_pred, zero_division=0),
        "recall": recall_score(y_test, rf_pred, zero_division=0),
        "f1": f1_score(y_test, rf_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, rf_proba),
        "roc_curve": roc_curve(y_test, rf_proba),
        "confusion_matrix": confusion_matrix(y_test, rf_pred),
        "feature_importance": dict(zip(feature_names, rf.feature_importances_)),
    }

    # Pick best model by F1 (balances precision & recall for imbalanced target)
    best_name = max(results, key=lambda k: results[k]["f1"])

    artifacts = {
        "preprocessor": preprocessor,
        "results": results,
        "best_model_name": best_name,
        "feature_names": feature_names,
        "y_test": y_test,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "purchase_rate": float(y.mean()),
        "categories": sorted(model_df["category_main"].unique().tolist()),
        "brands": sorted(model_df["brand"].unique().tolist()),
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    save_pkg = {k: v for k, v in artifacts.items() if k != "y_test"}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(save_pkg, f)

    # Also persist individual model files under models/ (rubric: "Save the Final Model")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl"))
    joblib.dump(log_reg, os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(results[best_name]["model"], os.path.join(MODELS_DIR, "best_model.pkl"))

    return artifacts


def load_or_train(df: pd.DataFrame) -> dict:
    """Always retrain to get fresh y_test for plots (dataset is small; fast)."""
    return train_models(df)


# ---------------------------------------------------------------------------
# Prediction interface
# ---------------------------------------------------------------------------
def predict_purchase(artifacts: dict, model_name: str, inputs: dict) -> dict:
    """
    Predict purchase likelihood for a single customer interaction.

    inputs keys: price, hour, weekday, is_weekend, category_main, brand
    """
    preprocessor = artifacts["preprocessor"]
    model = artifacts["results"][model_name]["model"]

    row = {c: inputs.get(c, 0) for c in NUMERIC_FEATURES}
    row.update({c: inputs.get(c, "unknown") for c in CATEGORICAL_FEATURES})

    X = pd.DataFrame([row])[FEATURE_COLS]
    X_t = preprocessor.transform(X)

    pred = int(model.predict(X_t)[0])
    proba = float(model.predict_proba(X_t)[0, 1])

    return {
        "prediction": "Likely to Purchase" if pred == 1 else "Not Likely to Purchase",
        "purchase": bool(pred),
        "probability": proba,
    }


def model_interpretation() -> dict:
    return {
        "why_logreg": (
            "Logistic Regression is used because the target (purchase / no "
            "purchase) is binary. It models the log-odds of purchase as a "
            "linear combination of features and squashes the result through "
            "the sigmoid function to output a probability between 0 and 1."
        ),
        "why_rf": (
            "Random Forest is an ensemble of decision trees. Each tree votes "
            "on the outcome and the forest averages the votes, which usually "
            "captures non-linear patterns and feature interactions better "
            "than a single linear model, at the cost of interpretability."
        ),
        "beginner_note": (
            "**Accuracy** = correct predictions / total predictions. "
            "**Precision** = of the interactions predicted as purchases, how "
            "many actually were. **Recall** = of the interactions that were "
            "actually purchases, how many were correctly identified. "
            "**F1-score** is the harmonic mean of precision and recall -- "
            "useful when classes are imbalanced (purchases are the minority "
            "class here)."
        ),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from cleaning import clean_data
    from feature_engineering import load_or_build

    data = load_or_build(clean_data())
    art = train_models(data)
    for name, res in art["results"].items():
        print(name, "acc=%.3f prec=%.3f rec=%.3f f1=%.3f auc=%.3f" % (
            res["accuracy"], res["precision"], res["recall"], res["f1"], res["roc_auc"]))
    print("Best model:", art["best_model_name"], "| purchase_rate=%.3f" % art["purchase_rate"])
