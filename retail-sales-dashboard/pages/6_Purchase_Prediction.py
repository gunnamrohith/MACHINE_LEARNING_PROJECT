"""
Page 6: Customer Purchase Prediction
=======================================
M7 - Binary Classification:
    - Train Logistic Regression + Random Forest on customer interaction data
    - Evaluate & compare: Accuracy, Precision, Recall, F1-Score
    - Confusion matrices for both models
    - Random Forest feature importance
    - Interactive "Will this customer purchase?" prediction interface
"""

import pandas as pd
import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cleaning import clean_data
from feature_engineering import load_or_build
from classification import (
    load_or_train, predict_purchase, model_interpretation,
    dataset_overview, detect_outliers_iqr, prepare_data,
)
from visualizations import (
    fig_purchase_distribution,
    fig_confusion_matrix,
    fig_model_comparison,
    fig_feature_importance,
    fig_roc_curve,
    fig_histogram,
    fig_boxplot,
    fig_scatter,
    fig_countplot,
    fig_correlation_heatmap,
)

st.set_page_config(page_title="Purchase Prediction", page_icon="🛒", layout="wide")


@st.cache_data(show_spinner=False)
def load_data():
    return load_or_build(clean_data())


@st.cache_resource(show_spinner="Training Logistic Regression & Random Forest…")
def get_models(df):
    return load_or_train(df)


df = load_data()
artifacts = get_models(df)
results = artifacts["results"]
interp = model_interpretation()

st.title("🛒 Customer Purchase Prediction")
st.markdown(
    "Predict whether a customer interaction is **Likely to Purchase** or "
    "**Not Likely to Purchase**, using **Logistic Regression** and "
    "**Random Forest** — trained and compared on the same data."
)

# ===========================================================================
# 1. Problem Definition
# ===========================================================================
with st.expander("📌 1. Problem Definition", expanded=False):
    st.markdown("""
- **Problem:** Predict whether a customer's interaction with a product will result in a
  **purchase** or **no purchase** — a binary classification problem.
- **Why it matters:** Most website visitors browse without buying. Knowing *who* is
  likely to purchase lets a business prioritise marketing spend, personalise offers,
  and reduce wasted outreach on low-intent visitors.
- **Who uses it:** E-commerce marketing/growth teams, recommendation engines, and
  CRM systems that decide who to target with discounts or reminders.
- **Expected output:** For a given customer interaction, the app outputs
  **"Likely to Purchase"** or **"Not Likely to Purchase"**, plus a purchase
  probability (0–100%).
""")

st.markdown("---")

# ===========================================================================
# 2 & 3. Dataset Collection + Data Understanding
# ===========================================================================
st.header("📂 2–3. Dataset Collection & Data Understanding")
st.markdown(
    "**Source:** *Ecommerce Behavior Data (Multi-Category Store)* schema "
    "([Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)). "
    "A `.csv` placed in `data/raw/` is used automatically; otherwise a synthetic "
    "sample of the same schema is generated. **Target column:** `event_type` "
    "(purchase vs. view/cart)."
)

overview = dataset_overview(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", f"{overview['n_records']:,}")
c2.metric("Features", overview["n_features"])
c3.metric("Duplicate Rows", overview["duplicate_rows"])
c4.metric("Missing Values (total)", sum(overview["missing_values"].values()))

with st.expander("🔍 Column details (dtypes, missing, unique values)"):
    col_info = pd.DataFrame({
        "dtype": overview["dtypes"],
        "missing": overview["missing_values"],
        "unique_values": overview["unique_values"],
    })
    st.dataframe(col_info, use_container_width=True)

with st.expander("📈 Statistical Summary (numeric columns)"):
    st.dataframe(overview["describe_numeric"], use_container_width=True)

st.markdown("---")

# ===========================================================================
# 4. Data Preprocessing (missing values, duplicates, outliers)
# ===========================================================================
st.header("🧹 4. Data Preprocessing")
st.markdown(
    "- **Missing values:** `brand`/`category_code` filled with `'unknown'`; rows "
    "with missing `price` dropped (handled in `src/cleaning.py`).\n"
    "- **Duplicates:** exact duplicate rows removed.\n"
    "- **Categorical variables:** `category_main`, `brand` → One-Hot Encoded.\n"
    "- **Numerical variables:** `price`, `hour`, `weekday`, `is_weekend` → Standard Scaled.\n"
    "- **Outlier check (IQR method on price):**"
)
outlier_info = detect_outliers_iqr(df, "price")
oc1, oc2, oc3 = st.columns(3)
oc1.metric("Outlier Bounds", f"₹{outlier_info['lower_bound']:.0f} – ₹{outlier_info['upper_bound']:.0f}")
oc2.metric("Outlier Rows", f"{outlier_info['n_outliers']:,}")
oc3.metric("% of Data", f"{outlier_info['pct_outliers']:.1f}%")
st.caption(
    "Outliers are kept (not removed) because high-value products are a normal "
    "part of e-commerce catalogues, not data errors; Random Forest is robust to them "
    "and StandardScaler still lets Logistic Regression handle the spread."
)

st.markdown("---")

# ===========================================================================
# 5. Exploratory Data Analysis
# ===========================================================================
st.header("📊 5. Exploratory Data Analysis")
model_df = prepare_data(df)

col_left, col_right = st.columns(2)
with col_left:
    st.plotly_chart(fig_histogram(df["price"], "💰 Price Distribution", "Price (₹)"),
                     use_container_width=True)
with col_right:
    st.plotly_chart(fig_countplot(df["event_type"], "🧮 Event Type Count", "Event Type"),
                     use_container_width=True)

col_left2, col_right2 = st.columns(2)
with col_left2:
    st.plotly_chart(fig_boxplot(df, "price", "event_type", "📦 Price by Event Type (Outlier View)"),
                     use_container_width=True)
with col_right2:
    st.plotly_chart(fig_scatter(df.sample(min(3000, len(df)), random_state=42),
                                 "hour", "price", "🕐 Price vs Hour of Day", color_col="event_type"),
                     use_container_width=True)

st.plotly_chart(
    fig_correlation_heatmap(model_df, ["price", "hour", "weekday", "is_weekend", "purchased"]),
    use_container_width=True
)
st.caption(
    "The correlation heatmap shows linear relationships between numeric features "
    "and the target. Low correlations here reflect the modest metrics below — "
    "expected for this synthetic sample dataset."
)

st.markdown("---")

# ===========================================================================
# 6. Feature Selection
# ===========================================================================
with st.expander("🧩 6. Feature Selection / Engineering", expanded=False):
    st.markdown("""
- **Used features:** `price`, `hour`, `weekday`, `is_weekend` (numeric) and
  `category_main`, `brand` (categorical) — all known *before* the purchase decision.
- **Removed:** raw IDs (`product_id`, `category_id`, `user_id`, `user_session`) —
  high-cardinality identifiers with no generalisable predictive value, and they
  risk the model memorising specific users/products instead of learning patterns.
- **Engineered:** `is_weekend` (derived from `weekday`) and `category_main`
  (top-level category extracted from the hierarchical `category_code`) — both
  created in `src/feature_engineering.py`.
""")

st.markdown("---")

# ===========================================================================
# 7. Train-Test Split & Target Overview
# ===========================================================================
st.header("✂️ 7. Train-Test Split & Target Distribution")
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Total Interactions", f"{artifacts['n_train'] + artifacts['n_test']:,}")
    st.metric("Purchase Rate", f"{artifacts['purchase_rate']:.1%}")
    st.metric("Train / Test Split (80/20)", f"{artifacts['n_train']:,} / {artifacts['n_test']:,}")
with col2:
    y_test = artifacts["y_test"]
    st.plotly_chart(fig_purchase_distribution(y_test.value_counts()), use_container_width=True)
st.caption("Split is stratified on the target with `random_state=42` for reproducibility.")

st.markdown("---")

# ===========================================================================
# 8-9. Model Building & Evaluation / Comparison
# ===========================================================================
st.header("⚖️ 8–9. Model Building, Evaluation & Comparison")

comparison_df = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Accuracy": results["logistic_regression"]["accuracy"],
        "Precision": results["logistic_regression"]["precision"],
        "Recall": results["logistic_regression"]["recall"],
        "F1-Score": results["logistic_regression"]["f1"],
        "ROC-AUC": results["logistic_regression"]["roc_auc"],
    },
    {
        "Model": "Random Forest",
        "Accuracy": results["random_forest"]["accuracy"],
        "Precision": results["random_forest"]["precision"],
        "Recall": results["random_forest"]["recall"],
        "F1-Score": results["random_forest"]["f1"],
        "ROC-AUC": results["random_forest"]["roc_auc"],
    },
])
st.dataframe(comparison_df.style.format({
    "Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
    "F1-Score": "{:.3f}", "ROC-AUC": "{:.3f}",
}), use_container_width=True)

best_name = artifacts["best_model_name"]
best_label = "Logistic Regression" if best_name == "logistic_regression" else "Random Forest"
st.success(f"🏆 **Best performing model (by F1-Score): {best_label}**")

st.plotly_chart(fig_model_comparison(comparison_df), use_container_width=True)

st.info(f"📖 **What do these metrics mean?** {interp['beginner_note']}")

st.markdown("---")

# ===========================================================================
# ROC Curve
# ===========================================================================
st.header("📈 ROC Curve")
roc_data = {
    "Logistic Regression": (*results["logistic_regression"]["roc_curve"][:2],
                             results["logistic_regression"]["roc_auc"]),
    "Random Forest": (*results["random_forest"]["roc_curve"][:2],
                       results["random_forest"]["roc_auc"]),
}
st.plotly_chart(fig_roc_curve(roc_data), use_container_width=True)
st.caption(
    "ROC-AUC measures how well the model ranks purchases above non-purchases "
    "across all thresholds. 0.5 = random guessing, 1.0 = perfect separation."
)

st.markdown("---")

# ===========================================================================
# Confusion Matrices
# ===========================================================================
st.header("🧩 Confusion Matrices")
col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(
        fig_confusion_matrix(results["logistic_regression"]["confusion_matrix"],
                              "Logistic Regression"),
        use_container_width=True
    )
with col_b:
    st.plotly_chart(
        fig_confusion_matrix(results["random_forest"]["confusion_matrix"],
                              "Random Forest"),
        use_container_width=True
    )
st.caption(
    "Rows = actual class, Columns = predicted class. The diagonal cells are "
    "correct predictions; off-diagonal cells are errors."
)

st.markdown("---")

# ===========================================================================
# Feature Importance
# ===========================================================================
st.header("🌲 Feature Importance (Random Forest)")
st.plotly_chart(
    fig_feature_importance(results["random_forest"]["feature_importance"]),
    use_container_width=True
)
st.caption(
    "These are the features the Random Forest found most useful for "
    "distinguishing purchases from non-purchases. This indicates predictive "
    "importance only, not a causal relationship."
)

with st.expander("🔍 Why Logistic Regression? Why Random Forest?"):
    st.markdown(f"**Logistic Regression:** {interp['why_logreg']}")
    st.markdown(f"**Random Forest:** {interp['why_rf']}")

st.markdown("---")

# ===========================================================================
# Interactive Prediction Interface
# ===========================================================================
st.header("🔮 10. Make a Prediction (Deployed Model in Action)")
st.markdown("Enter details of a customer interaction to get a live prediction.")

categories = artifacts.get("categories", ["electronics", "computers", "appliances"])
brands = artifacts.get("brands", ["samsung", "apple", "xiaomi"])

col1, col2 = st.columns(2)
with col1:
    selected_cat = st.selectbox("📂 Product Category", options=categories)
    selected_brand = st.selectbox("🏷️ Brand", options=brands)
    price = st.number_input("💵 Product Price (₹)", min_value=0.0, value=100.0, step=10.0)

with col2:
    hour = st.slider("🕐 Hour of Day", 0, 23, 12)
    weekday = st.selectbox(
        "📅 Day of Week",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday",
                                "Friday", "Saturday", "Sunday"][x]
    )
    model_choice = st.radio(
        "🤖 Model to use",
        options=["logistic_regression", "random_forest"],
        format_func=lambda x: "Logistic Regression" if x == "logistic_regression" else "Random Forest",
        index=0 if best_name == "logistic_regression" else 1,
        horizontal=True,
    )

if st.button("🚀 Predict Purchase", type="primary"):
    inputs = {
        "price": price,
        "hour": hour,
        "weekday": weekday,
        "is_weekend": 1 if weekday in (5, 6) else 0,
        "category_main": selected_cat,
        "brand": selected_brand,
    }
    result = predict_purchase(artifacts, model_choice, inputs)

    if result["purchase"]:
        st.success(f"### ✅ {result['prediction']}")
    else:
        st.error(f"### ❌ {result['prediction']}")

    st.markdown(f"**Purchase Probability: {result['probability']:.1%}**")
    st.progress(min(max(result["probability"], 0.0), 1.0))
    st.caption(
        f"Prediction made using **{'Logistic Regression' if model_choice=='logistic_regression' else 'Random Forest'}**, "
        f"trained on {artifacts['n_train']:,} interactions."
    )

st.markdown("---")
st.caption("M7: Purchase Prediction | Logistic Regression · Random Forest · scikit-learn")

# ===========================================================================
# 10 (persistence) & 11. Save the Final Model + Deployment
# ===========================================================================
st.markdown("---")
st.header("💾 10–11. Model Persistence & Deployment")
st.markdown(f"""
**Save the Final Model:** every training run saves the fitted preprocessing pipeline
and both models to disk with `joblib`, so the app never retrains at prediction time:

```
models/
├── preprocessing_pipeline.pkl   ← ColumnTransformer (scaler + one-hot encoder)
├── logistic_regression.pkl
├── random_forest.pkl
└── best_model.pkl               ← currently: {best_label}
```

**Deployment:** this Streamlit app is deployment-ready as-is (single entry point
`app.py`, all dependencies pinned in `requirements.txt`, no external services
required). To deploy it live on **Streamlit Community Cloud** (free):

1. Push this project folder to a public/private GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select the repo, branch, and set the main file path to `app.py`.
4. Click **Deploy** — Streamlit Cloud installs `requirements.txt` and starts the app.
5. Share the generated `https://<app-name>.streamlit.app` link.

Alternatively, for a container-based deployment (Render, Railway, Hugging Face
Spaces, etc.), the same `requirements.txt` + `streamlit run app.py --server.port $PORT
--server.address 0.0.0.0` command works unchanged.
""")
