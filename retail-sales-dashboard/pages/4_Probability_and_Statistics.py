"""
Page 4: Probability & Statistics
==================================
M3 – Probability Analysis:
    - P(View), P(Cart), P(Purchase)
    - P(Purchase | View), P(Purchase | Cart)
    - Conversion Funnel

M4 – Statistical Inference:
    - 95% Confidence Interval for average price
    - Weekend vs Weekday T-Test
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cleaning import clean_data
from feature_engineering import load_or_build
from probability import (
    event_probabilities, conditional_probabilities, funnel_analysis,
    interpret_purchase_given_view, interpret_funnel
)
from statistics import (
    confidence_interval_price, interpret_ci,
    ttest_weekend_vs_weekday, interpret_ttest
)
from visualizations import fig_event_probability_bar, fig_funnel

st.set_page_config(page_title="Probability & Statistics", page_icon="🎲", layout="wide")


@st.cache_data(show_spinner=False)
def load_data():
    return load_or_build(clean_data())


df = load_data()

st.title("🎲 Probability & Statistics")
st.markdown(
    "Apply probability theory and statistical inference to uncover customer conversion behaviour."
)
st.markdown("---")

# ===========================================================================
# M3 – PROBABILITY
# ===========================================================================
st.header("📐 M3: Probability Analysis")

# --- Event Probabilities ---
st.subheader("1️⃣ Event Probabilities")
probs = event_probabilities(df)

c1, c2, c3 = st.columns(3)
c1.metric("P(View)", f"{probs['P(view)']:.1%}")
c2.metric("P(Cart)", f"{probs['P(cart)']:.1%}")
c3.metric("P(Purchase)", f"{probs['P(purchase)']:.1%}")

col_fig, col_explain = st.columns([2, 1])
with col_fig:
    st.plotly_chart(fig_event_probability_bar(probs), use_container_width=True)
with col_explain:
    st.markdown("### What does this mean?")
    st.markdown(
        f"- **{probs['P(view)']:.1%}** of all events are product views.\n"
        f"- **{probs['P(cart)']:.1%}** of all events are cart additions.\n"
        f"- Only **{probs['P(purchase)']:.1%}** of all events lead to a purchase.\n\n"
        "This shows that the store has strong browsing activity but a relatively "
        "low purchase rate — improving conversion is key."
    )

st.markdown("---")

# --- Conditional Probabilities ---
st.subheader("2️⃣ Conditional Probabilities")
cond = conditional_probabilities(df)

c1, c2 = st.columns(2)
with c1:
    st.metric("P(Purchase | View)", f"{cond['P(Purchase | View)']:.1%}")
    st.markdown(interpret_purchase_given_view(cond["P(Purchase | View)"]))
with c2:
    st.metric("P(Purchase | Cart)", f"{cond['P(Purchase | Cart)']:.1%}")
    st.markdown(
        f"Out of every 100 products added to cart, approximately "
        f"**{cond['P(Purchase | Cart)'] * 100:.1f}** are eventually purchased. "
        "Cart-abandonment is a major opportunity — retargeting emails can recover this."
    )

st.markdown("---")

# --- Funnel Analysis ---
st.subheader("3️⃣ Conversion Funnel")
funnel = funnel_analysis(df)

col_fig, col_explain = st.columns([1, 1])
with col_fig:
    st.plotly_chart(fig_funnel(funnel), use_container_width=True)
with col_explain:
    st.markdown("### Funnel Rates")
    st.markdown(
        f"| Stage | Rate |\n"
        f"|-------|------|\n"
        f"| View → Cart | **{funnel['view_to_cart_rate']:.1%}** |\n"
        f"| Cart → Purchase | **{funnel['cart_to_purchase_rate']:.1%}** |\n"
        f"| Overall (View → Purchase) | **{funnel['overall_conversion_rate']:.1%}** |"
    )
    st.markdown("")
    st.info(interpret_funnel(funnel))

st.markdown("---")

# ===========================================================================
# M4 – STATISTICAL INFERENCE
# ===========================================================================
st.header("📏 M4: Statistical Inference")

# --- Confidence Interval ---
st.subheader("1️⃣ 95% Confidence Interval — Average Product Price")
ci = confidence_interval_price(df)

c1, c2, c3 = st.columns(3)
c1.metric("Sample Mean", f"₹{ci['mean']:.2f}")
c2.metric("Lower Bound (95%)", f"₹{ci['lower']:.2f}")
c3.metric("Upper Bound (95%)", f"₹{ci['upper']:.2f}")

st.success(interpret_ci(ci))
with st.expander("📖 How is this calculated?"):
    st.markdown(
        "We use the **t-distribution** confidence interval formula:\n\n"
        "```\nCI = x̄ ± t* × (s / √n)\n```\n\n"
        f"- **x̄** (sample mean) = ₹{ci['mean']:.2f}\n"
        f"- **s** (std dev) = ₹{ci['std']:.2f}\n"
        f"- **n** (sample size) = {ci['n']:,}\n"
        "- **t\\*** = critical value from t-distribution at 95% confidence\n\n"
        "A narrower interval means we're more certain about the true average. "
        "Large sample sizes produce narrower intervals."
    )

st.markdown("---")

# --- Hypothesis Test ---
st.subheader("2️⃣ Hypothesis Test — Weekend vs Weekday Purchases")
result = ttest_weekend_vs_weekday(df)

with st.expander("📋 Hypothesis Setup"):
    st.markdown(
        f"**H₀ (Null Hypothesis):** {result['h0']}\n\n"
        f"**H₁ (Alternative Hypothesis):** {result['h1']}\n\n"
        f"**Test:** Independent Samples Welch's T-Test (two-tailed)\n\n"
        f"**Significance Level (α):** {result['alpha']}"
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Weekday Mean", f"₹{result['weekday_mean']:.2f}")
c2.metric("Weekend Mean", f"₹{result['weekend_mean']:.2f}")
c3.metric("T-Statistic", f"{result['t_statistic']:.4f}")
c4.metric("P-Value", f"{result['p_value']:.4f}")

if result["p_value"] < result["alpha"]:
    st.error(f"⚠️ **Conclusion:** {result['conclusion'].upper()} — Result is statistically significant.")
else:
    st.success(f"✅ **Conclusion:** {result['conclusion'].upper()} — No significant difference found.")

st.info(interpret_ttest(result))

st.markdown("---")
st.caption("M3: Probability Analysis | M4: Statistical Inference")
