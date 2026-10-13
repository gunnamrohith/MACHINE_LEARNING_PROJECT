"""
Retail Sales Analytics Dashboard
==================================
Home Page – KPI Cards + Revenue Overview

Run: streamlit run app.py
"""

import streamlit as st
import sys
import os

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from cleaning import clean_data
from feature_engineering import load_or_build
from eda import kpi_summary, revenue_trend_daily, revenue_by_category
from eda_report import build_eda_pdf
from visualizations import fig_revenue_trend, fig_revenue_by_category

EDA_REPORT_VERSION = "2.0"

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Retail Sales Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading & cleaning data…")
def load_data():
    cleaned = clean_data()
    featured = load_or_build(cleaned)
    return featured


@st.cache_data(show_spinner="Preparing EDA report...")
def generate_eda_report(data, report_version):
    return build_eda_pdf(data)


df = load_data()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.image(
    "https://img.icons8.com/color/96/shopping-cart.png", width=60
)
st.sidebar.title("🛍️ Retail Analytics")
st.sidebar.markdown("**Navigate using the pages below.**")
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Info**")
st.sidebar.markdown(f"- Rows: `{len(df):,}`")
st.sidebar.markdown(f"- Columns: `{df.shape[1]}`")
st.sidebar.markdown(f"- Date range: `{df['event_time'].dt.date.min()}` → `{df['event_time'].dt.date.max()}`")


# ---------------------------------------------------------------------------
# Home content
# ---------------------------------------------------------------------------
st.title("🛍️ Retail Sales Analytics Dashboard")
st.markdown(
    "Welcome! This dashboard walks through a **complete data analytics lifecycle** — "
    "from raw data cleaning to predictive modelling — for an e-commerce behaviour dataset."
)
st.markdown("---")

# --- KPI Cards ---
kpi = kpi_summary(df)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("💰 Total Revenue", f"₹{kpi['total_revenue']:,.0f}")
with col2:
    st.metric("👥 Total Customers", f"{kpi['total_customers']:,}")
with col3:
    st.metric("📦 Total Products", f"{kpi['total_products']:,}")
with col4:
    st.metric("🛒 Total Purchases", f"{kpi['total_purchases']:,}")

st.markdown("---")

# --- Charts ---
col_left, col_right = st.columns([2, 1])

with col_left:
    daily = revenue_trend_daily(df)
    st.plotly_chart(fig_revenue_trend(daily), use_container_width=True)
    st.caption(
        "💡 **Business Insight**: Identify peak sales periods to plan inventory and "
        "marketing campaigns. Dips may indicate seasonal slow-downs or data gaps."
    )

with col_right:
    cat = revenue_by_category(df)
    st.plotly_chart(fig_revenue_by_category(cat), use_container_width=True)
    st.caption(
        "💡 **Business Insight**: The top 2–3 categories often drive 80% of revenue "
        "(Pareto principle). Focus supplier negotiations on these categories."
    )

st.markdown("---")
st.markdown("### Download EDA Report")
st.markdown(
    "Download a complete PDF summary covering data quality, event distribution, "
    "sales trends, category performance, customer and product analysis, and "
    "business recommendations. Version 2.0 includes event and category pie charts "
    "plus predictive price-model metrics and an actual-vs-predicted chart."
)
st.download_button(
    label="Download EDA Report v2.0 (PDF)",
    data=generate_eda_report(df, EDA_REPORT_VERSION),
    file_name="retail_sales_eda_report_v2.pdf",
    mime="application/pdf",
    type="primary",
)

st.markdown("---")
st.markdown("### 📚 Project Modules")
cols = st.columns(5)
modules = [
    ("📊", "Sales Analysis", "pages/1_Sales_Analysis.py"),
    ("👥", "Customer Analysis", "pages/2_Customer_Analysis.py"),
    ("📦", "Product Analysis", "pages/3_Product_Analysis.py"),
    ("🎲", "Probability & Stats", "pages/4_Probability_and_Statistics.py"),
    ("🤖", "Predictive Analytics", "pages/5_Predictive_Analytics.py"),
]
for col, (icon, title, _) in zip(cols, modules):
    col.info(f"{icon} **{title}**")

st.markdown("---")
st.caption("Built with Python · Pandas · Plotly · Streamlit · SciPy · Scikit-Learn")
