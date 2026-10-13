"""
Page 2: Customer Analysis
==========================
Covers:
- Total unique customers
- Top customers by spend
- Customer activity distribution
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cleaning import clean_data
from feature_engineering import load_or_build
from eda import total_customers, top_customers, customer_activity_distribution
from visualizations import fig_customer_activity, fig_top_customers

st.set_page_config(page_title="Customer Analysis", page_icon="👥", layout="wide")


@st.cache_data(show_spinner=False)
def load_data():
    return load_or_build(clean_data())


df = load_data()

st.title("👥 Customer Analysis")
st.markdown("Understand who your customers are and how they interact with your store.")
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------------------------
st.sidebar.header("🔍 Filters")
min_date = df["event_time"].dt.date.min()
max_date = df["event_time"].dt.date.max()
date_range = st.sidebar.date_input(
    "Date Range", value=(min_date, max_date),
    min_value=min_date, max_value=max_date, key="cust_date"
)
top_n = st.sidebar.slider("Top N Customers", min_value=5, max_value=20, value=10)

filtered = df.copy()
if len(date_range) == 2:
    s, e = date_range
    filtered = filtered[
        (filtered["event_time"].dt.date >= s) &
        (filtered["event_time"].dt.date <= e)
    ]

# ---------------------------------------------------------------------------
# KPI Row
# ---------------------------------------------------------------------------
n_customers = total_customers(filtered)
purchases = filtered[filtered["event_type"] == "purchase"]
n_buyers = purchases["user_id"].nunique()
repeat_rate = (
    (purchases.groupby("user_id").size() > 1).sum() / n_buyers * 100
    if n_buyers > 0 else 0
)

c1, c2, c3 = st.columns(3)
c1.metric("👥 Unique Visitors", f"{n_customers:,}")
c2.metric("💳 Unique Buyers", f"{n_buyers:,}")
c3.metric("🔄 Repeat Buyer Rate", f"{repeat_rate:.1f}%")

st.markdown("---")

# ---------------------------------------------------------------------------
# Customer Activity Distribution
# ---------------------------------------------------------------------------
st.subheader("📊 Customer Activity Distribution")
activity = customer_activity_distribution(filtered)
st.plotly_chart(fig_customer_activity(activity), use_container_width=True)
st.info(
    f"💡 **Insight**: Most customers have fewer than {int(activity.median())} interactions. "
    "This right-skewed distribution is typical in e-commerce — a small group of "
    "highly active users drives most engagement. Focus loyalty programmes on them."
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Top Customers
# ---------------------------------------------------------------------------
st.subheader(f"🏆 Top {top_n} Customers by Spend")
top_df = top_customers(filtered, n=top_n)
col_chart, col_table = st.columns([2, 1])
with col_chart:
    st.plotly_chart(fig_top_customers(top_df), use_container_width=True)
with col_table:
    display_df = top_df.copy()
    display_df["user_id"] = "User …" + display_df["user_id"].astype(str).str[-5:]
    display_df = display_df.rename(columns={
        "user_id": "Customer", "total_spend": "Total Spend (₹)", "num_purchases": "Orders"
    })
    display_df["Total Spend (₹)"] = display_df["Total Spend (₹)"].map("₹{:,.2f}".format)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.info(
    "💡 **Insight**: These VIP customers represent a disproportionate share of "
    "revenue. Targeted retention strategies (exclusive offers, early access) can "
    "significantly protect and grow total revenue."
)

st.markdown("---")
st.caption("M2: Exploratory Data Analysis | Customer Module")
