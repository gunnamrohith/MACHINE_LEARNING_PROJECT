"""
Page 5: Predictive Analytics
==============================
M5 – Linear Regression:
    - Train model on purchase events
    - Evaluate: MAE, RMSE, R²
    - Interactive prediction interface
    - Actual vs Predicted + Residual plots
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cleaning import clean_data
from feature_engineering import load_or_build
from regression import load_or_train, predict_price, model_interpretation
from visualizations import fig_actual_vs_predicted, fig_residuals

st.set_page_config(page_title="Predictive Analytics", page_icon="🤖", layout="wide")


@st.cache_data(show_spinner=False)
def load_data():
    return load_or_build(clean_data())


@st.cache_resource(show_spinner="Training model…")
def get_model(df):
    return load_or_train(df)


df = load_data()
model_artifacts = get_model(df)

st.title("🤖 Predictive Analytics")
st.markdown(
    "Build and evaluate a **Linear Regression** model to predict product price "
    "based on category, brand, hour of day, and day of week."
)
st.markdown("---")

# ===========================================================================
# Model Evaluation
# ===========================================================================
st.header("📊 Model Performance")

c1, c2, c3 = st.columns(3)
c1.metric("MAE (Mean Absolute Error)", f"₹{model_artifacts['mae']:.2f}",
          help="Average rupee error in predictions")
c2.metric("RMSE (Root Mean Squared Error)", f"₹{model_artifacts['rmse']:.2f}",
          help="Penalises large errors more than MAE")
c3.metric("R² Score", f"{model_artifacts['r2']:.4f}",
          help="Fraction of price variation explained by the model (0–1)")

interp = model_interpretation()
st.info(f"📖 **What do these metrics mean?** {interp['beginner_note']}")

st.markdown("---")

# ===========================================================================
# Visualisations
# ===========================================================================
st.header("📈 Model Visualisations")
col_left, col_right = st.columns(2)

with col_left:
    st.plotly_chart(
        fig_actual_vs_predicted(model_artifacts["y_test"], model_artifacts["y_pred"]),
        use_container_width=True
    )
    st.caption(
        "Points close to the red dashed line = accurate predictions. "
        "Spread indicates prediction uncertainty."
    )

with col_right:
    st.plotly_chart(
        fig_residuals(model_artifacts["y_test"], model_artifacts["y_pred"]),
        use_container_width=True
    )
    st.caption(
        "Residuals should be randomly scattered around zero. "
        "Patterns in residuals suggest the model is missing important features."
    )

st.markdown("---")

# ===========================================================================
# Strengths & Limitations
# ===========================================================================
with st.expander("🔍 Model Strengths & Limitations"):
    col_s, col_l = st.columns(2)
    with col_s:
        st.markdown("### ✅ Strengths")
        for s in interp["strengths"]:
            st.markdown(f"- {s}")
    with col_l:
        st.markdown("### ⚠️ Limitations")
        for l in interp["limitations"]:
            st.markdown(f"- {l}")

st.markdown("---")

# ===========================================================================
# Interactive Prediction Interface
# ===========================================================================
st.header("🔮 Make a Prediction")
st.markdown("Adjust the inputs below to predict the expected product price.")

categories = model_artifacts.get("categories", ["electronics", "computers", "appliances"])
brands = model_artifacts.get("brands", ["samsung", "apple", "xiaomi"])

col1, col2 = st.columns(2)
with col1:
    selected_cat = st.selectbox("📂 Product Category", options=categories)
    selected_brand = st.selectbox("🏷️ Brand", options=brands)

with col2:
    hour = st.slider("🕐 Hour of Day", 0, 23, 12)
    weekday = st.selectbox(
        "📅 Day of Week",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday",
                                "Friday", "Saturday", "Sunday"][x]
    )

if st.button("🚀 Predict Price", type="primary"):
    predicted = predict_price(model_artifacts, selected_cat, selected_brand, hour, weekday)
    st.success(f"### 💰 Predicted Price: ₹{predicted:,.2f}")
    st.markdown(
        f"The model estimates a product in the **{selected_cat}** category "
        f"from **{selected_brand}**, purchased at **{hour:02d}:00** on a "
        f"**{['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][weekday]}**, "
        f"would be priced around **₹{predicted:,.2f}**."
    )
    st.caption(
        f"⚠️ This is an estimate with a ±₹{model_artifacts['mae']:.2f} average error. "
        "Use it as a starting point, not an exact figure."
    )

st.markdown("---")
st.caption("M5: Regression Analysis | Linear Regression · scikit-learn")
