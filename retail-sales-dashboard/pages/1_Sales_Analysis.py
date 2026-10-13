"""
Page 1: Sales Analysis
========================
Covers:
- Revenue by Category (filtered)
- Revenue Trend (filtered by date range)
- Monthly Revenue
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cleaning import clean_data
from feature_engineering import load_or_build
from eda import revenue_by_category, revenue_trend_daily, revenue_by_month
from visualizations import (
    fig_revenue_by_category, fig_revenue_trend, fig_monthly_revenue
)

st.set_page_config(page_title="Sales Analysis", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def load_data():
    return load_or_build(clean_data())


df = load_data()

st.title("📊 Sales Analysis")
st.markdown("Explore revenue patterns across categories, time, and months.")
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------------------------
st.sidebar.header("🔍 Filters")

all_categories = sorted(df["category_main"].dropna().unique().tolist())
selected_cats = st.sidebar.multiselect(
    "Category", all_categories, default=all_categories, key="sales_cat"
)

min_date = df["event_time"].dt.date.min()
max_date = df["event_time"].dt.date.max()
date_range = st.sidebar.date_input(
    "Date Range", value=(min_date, max_date),
    min_value=min_date, max_value=max_date, key="sales_date"
)

# Apply filters
filtered = df[df["category_main"].isin(selected_cats)]
if len(date_range) == 2:
    start, end = date_range
    filtered = filtered[
        (filtered["event_time"].dt.date >= start) &
        (filtered["event_time"].dt.date <= end)
    ]

purchases = filtered[filtered["event_type"] == "purchase"]

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
total_rev = purchases["price"].sum()
num_orders = len(purchases)
avg_order = purchases["price"].mean() if num_orders > 0 else 0

c1, c2, c3 = st.columns(3)
c1.metric("💰 Revenue (Filtered)", f"₹{total_rev:,.0f}")
c2.metric("🛒 Orders", f"{num_orders:,}")
c3.metric("📦 Avg Order Value", f"₹{avg_order:,.2f}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Revenue by Category
# ---------------------------------------------------------------------------
st.subheader("🏷️ Revenue by Category")
cat_df = revenue_by_category(filtered)
if not cat_df.empty:
    st.plotly_chart(fig_revenue_by_category(cat_df), use_container_width=True)
    top = cat_df.iloc[0]
    st.info(
        f"💡 **Insight**: **{top['category_main'].title()}** is the top revenue-generating "
        f"category with ₹{top['revenue']:,.0f}. "
        "Consider prioritising stock, ads, and promotions for this category."
    )
else:
    st.warning("No purchase data available for the selected filters.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Revenue Trend
# ---------------------------------------------------------------------------
st.subheader("📈 Revenue Trend (Daily)")
daily_df = revenue_trend_daily(filtered)
if not daily_df.empty:
    st.plotly_chart(fig_revenue_trend(daily_df), use_container_width=True)
    peak_day = daily_df.loc[daily_df["revenue"].idxmax()]
    st.info(
        f"💡 **Insight**: Peak revenue was recorded on **{peak_day['date']}** "
        f"with ₹{peak_day['revenue']:,.0f}. "
        "Analyse what drove this spike (promotions, pay-day cycles, etc.)."
    )
else:
    st.warning("No data for the selected filters.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Monthly Revenue
# ---------------------------------------------------------------------------
st.subheader("📅 Monthly Revenue")
monthly_df = revenue_by_month(filtered)
if not monthly_df.empty:
    st.plotly_chart(fig_monthly_revenue(monthly_df), use_container_width=True)
    best_month = monthly_df.loc[monthly_df["revenue"].idxmax()]
    st.info(
        f"💡 **Insight**: **{best_month['period']}** was the best-performing month "
        f"with ₹{best_month['revenue']:,.0f} in revenue. "
        "Use this to forecast and plan capacity for similar future months."
    )

st.markdown("---")
st.caption("M2: Exploratory Data Analysis | Sales Module")
