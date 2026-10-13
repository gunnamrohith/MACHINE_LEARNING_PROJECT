"""
Page 3: Product Analysis
=========================
Covers:
- Most viewed products
- Most purchased products
- Category popularity (pie chart)
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cleaning import clean_data
from feature_engineering import load_or_build
from eda import most_viewed_products, most_purchased_products, category_popularity, total_products
from visualizations import fig_top_viewed, fig_top_purchased, fig_category_pie

st.set_page_config(page_title="Product Analysis", page_icon="📦", layout="wide")


@st.cache_data(show_spinner=False)
def load_data():
    return load_or_build(clean_data())


df = load_data()

st.title("📦 Product Analysis")
st.markdown("Discover which products and categories drive the most interest and sales.")
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------------------------
st.sidebar.header("🔍 Filters")
all_cats = sorted(df["category_main"].dropna().unique().tolist())
selected_cats = st.sidebar.multiselect(
    "Category", all_cats, default=all_cats, key="prod_cat"
)
top_n = st.sidebar.slider("Top N Products", 5, 20, 10)

filtered = df[df["category_main"].isin(selected_cats)]

# ---------------------------------------------------------------------------
# KPI Row
# ---------------------------------------------------------------------------
n_products = total_products(filtered)
n_views = len(filtered[filtered["event_type"] == "view"])
n_purchases = len(filtered[filtered["event_type"] == "purchase"])

c1, c2, c3 = st.columns(3)
c1.metric("📦 Unique Products", f"{n_products:,}")
c2.metric("👁️ Total Views", f"{n_views:,}")
c3.metric("🛒 Total Purchases", f"{n_purchases:,}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Top Viewed & Purchased side-by-side
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("👁️ Most Viewed Products")
    viewed_df = most_viewed_products(filtered, n=top_n)
    st.plotly_chart(fig_top_viewed(viewed_df), use_container_width=True)
    st.info(
        "💡 **Insight**: High-view, low-purchase products suffer from a conversion problem. "
        "Investigate pricing, descriptions, or images for these items."
    )

with col_right:
    st.subheader("🛒 Most Purchased Products")
    purchased_df = most_purchased_products(filtered, n=top_n)
    st.plotly_chart(fig_top_purchased(purchased_df), use_container_width=True)
    st.info(
        "💡 **Insight**: Best-sellers drive revenue. Ensure adequate stock, "
        "prioritise them in search results, and use them as upsell anchors."
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Category Popularity
# ---------------------------------------------------------------------------
st.subheader("🥧 Category Popularity")
cat_df = category_popularity(filtered)
col_pie, col_bar = st.columns([1, 1])
with col_pie:
    st.plotly_chart(fig_category_pie(cat_df), use_container_width=True)
with col_bar:
    st.dataframe(
        cat_df.rename(columns={"category_main": "Category", "event_count": "Total Events"})
              .style.bar(subset=["Total Events"], color="#636EFA"),
        use_container_width=True,
        hide_index=True,
    )
st.info(
    "💡 **Insight**: A small number of categories attract the majority of traffic. "
    "Expand product range within high-traffic categories to capture more wallet share."
)

st.markdown("---")
st.caption("M2: Exploratory Data Analysis | Product Module")
