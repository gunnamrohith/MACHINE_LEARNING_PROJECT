"""
M6: Visualizations Module
===========================
Centralised Plotly figure builders used across all Streamlit pages.
Every function returns a go.Figure / px figure ready for st.plotly_chart().
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
PALETTE = px.colors.qualitative.Plotly
ACCENT = "#636EFA"


# ---------------------------------------------------------------------------
# Sales charts
# ---------------------------------------------------------------------------
def fig_revenue_trend(daily_df: pd.DataFrame) -> go.Figure:
    """Line chart: daily revenue over time."""
    fig = px.line(
        daily_df, x="date", y="revenue",
        title="📈 Revenue Trend (Daily)",
        labels={"date": "Date", "revenue": "Revenue (₹)"},
        color_discrete_sequence=[ACCENT],
    )
    fig.update_traces(mode="lines", line_width=1.5)
    fig.update_layout(hovermode="x unified")
    return fig


def fig_revenue_by_category(cat_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar: revenue by category."""
    fig = px.bar(
        cat_df.sort_values("revenue"),
        x="revenue", y="category_main",
        orientation="h",
        title="🏷️ Revenue by Category",
        labels={"revenue": "Revenue (₹)", "category_main": "Category"},
        color="revenue",
        color_continuous_scale="Blues",
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    return fig


def fig_monthly_revenue(monthly_df: pd.DataFrame) -> go.Figure:
    """Bar chart: monthly revenue."""
    fig = px.bar(
        monthly_df, x="period", y="revenue",
        title="📅 Monthly Revenue",
        labels={"period": "Month", "revenue": "Revenue (₹)"},
        color_discrete_sequence=[ACCENT],
    )
    fig.update_xaxes(tickangle=45)
    return fig


# ---------------------------------------------------------------------------
# Customer charts
# ---------------------------------------------------------------------------
def fig_customer_activity(activity_series: pd.Series) -> go.Figure:
    """Histogram: distribution of events per customer."""
    fig = px.histogram(
        x=activity_series.values,
        nbins=50,
        title="👥 Customer Activity Distribution",
        labels={"x": "Events per Customer", "y": "Number of Customers"},
        color_discrete_sequence=[ACCENT],
    )
    fig.update_layout(bargap=0.05)
    return fig


def fig_top_customers(top_df: pd.DataFrame) -> go.Figure:
    """Bar chart: top N customers by spend."""
    top_df = top_df.copy()
    top_df["user_label"] = "User " + top_df["user_id"].astype(str).str[-5:]
    fig = px.bar(
        top_df.sort_values("total_spend"),
        x="total_spend", y="user_label",
        orientation="h",
        title="🏆 Top Customers by Spend",
        labels={"total_spend": "Total Spend (₹)", "user_label": "Customer"},
        color="total_spend",
        color_continuous_scale="Oranges",
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


# ---------------------------------------------------------------------------
# Product charts
# ---------------------------------------------------------------------------
def fig_top_viewed(df: pd.DataFrame) -> go.Figure:
    """Bar chart: top viewed products."""
    df = df.copy()
    df["product_label"] = "Product " + df["product_id"].astype(str).str[-5:]
    fig = px.bar(
        df.sort_values("view_count"),
        x="view_count", y="product_label",
        orientation="h",
        title="👁️ Top Viewed Products",
        labels={"view_count": "Views", "product_label": "Product"},
        color_discrete_sequence=["#EF553B"],
    )
    return fig


def fig_top_purchased(df: pd.DataFrame) -> go.Figure:
    """Bar chart: top purchased products."""
    df = df.copy()
    df["product_label"] = "Product " + df["product_id"].astype(str).str[-5:]
    fig = px.bar(
        df.sort_values("purchase_count"),
        x="purchase_count", y="product_label",
        orientation="h",
        title="🛒 Top Purchased Products",
        labels={"purchase_count": "Purchases", "product_label": "Product"},
        color_discrete_sequence=["#00CC96"],
    )
    return fig


def fig_category_pie(cat_df: pd.DataFrame) -> go.Figure:
    """Pie chart: category share of total events."""
    fig = px.pie(
        cat_df.head(8),
        names="category_main",
        values="event_count",
        title="🥧 Category Popularity (Event Share)",
        color_discrete_sequence=PALETTE,
        hole=0.3,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


# ---------------------------------------------------------------------------
# Probability & Funnel
# ---------------------------------------------------------------------------
def fig_event_probability_bar(probs: dict) -> go.Figure:
    """Bar chart of marginal event probabilities."""
    labels = ["P(View)", "P(Cart)", "P(Purchase)"]
    values = [probs.get(l, 0) for l in labels]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=["#636EFA", "#EF553B", "#00CC96"],
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="📊 Event Probability Distribution",
        yaxis=dict(tickformat=".0%", title="Probability"),
        xaxis_title="Event Type",
    )
    return fig


def fig_funnel(funnel: dict) -> go.Figure:
    """Plotly funnel chart for the conversion funnel."""
    fig = go.Figure(go.Funnel(
        y=funnel["stages"],
        x=funnel["counts"],
        textinfo="value+percent initial",
        marker_color=["#636EFA", "#AB63FA", "#00CC96"],
    ))
    fig.update_layout(title="🔽 Conversion Funnel: View → Cart → Purchase")
    return fig


# ---------------------------------------------------------------------------
# Regression charts
# ---------------------------------------------------------------------------
def fig_actual_vs_predicted(y_test, y_pred) -> go.Figure:
    """Scatter: actual vs predicted prices."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred,
        mode="markers",
        marker=dict(color=ACCENT, opacity=0.4, size=4),
        name="Predictions",
    ))
    # Perfect prediction line
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", line=dict(color="red", dash="dash"),
        name="Perfect Prediction",
    ))
    fig.update_layout(
        title="🎯 Actual vs Predicted Price",
        xaxis_title="Actual Price (₹)",
        yaxis_title="Predicted Price (₹)",
    )
    return fig


def fig_residuals(y_test, y_pred) -> go.Figure:
    """Residual plot."""
    residuals = y_test - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals,
        mode="markers",
        marker=dict(color="#EF553B", opacity=0.4, size=4),
        name="Residuals",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(
        title="📉 Residual Plot",
        xaxis_title="Predicted Price (₹)",
        yaxis_title="Residual (Actual − Predicted)",
    )
    return fig


# ---------------------------------------------------------------------------
# Classification charts (Purchase Prediction)
# ---------------------------------------------------------------------------
def fig_purchase_distribution(counts: pd.Series) -> go.Figure:
    """Bar chart: Purchase vs No Purchase class distribution."""
    labels = ["No Purchase", "Purchase"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=["#636EFA", "#00CC96"],
        text=values, textposition="outside",
    ))
    fig.update_layout(
        title="🎯 Purchase vs No Purchase Distribution",
        yaxis_title="Number of Interactions",
    )
    return fig


def fig_confusion_matrix(cm: np.ndarray, title: str) -> go.Figure:
    """Heatmap confusion matrix for a binary classifier."""
    labels = ["No Purchase", "Purchase"]
    fig = px.imshow(
        cm, x=labels, y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title=title,
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


def fig_model_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing Accuracy/Precision/Recall/F1 across models."""
    melted = comparison_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(
        melted, x="Metric", y="Score", color="Model",
        barmode="group",
        title="⚖️ Model Comparison: Logistic Regression vs Random Forest",
        color_discrete_sequence=["#636EFA", "#00CC96"],
        text_auto=".3f",
    )
    fig.update_layout(yaxis=dict(range=[0, 1], title="Score"))
    return fig


def fig_feature_importance(importance: dict, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart of top-N Random Forest feature importances."""
    items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [i[0] for i in items][::-1]
    vals = [i[1] for i in items][::-1]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color="#AB63FA",
    ))
    fig.update_layout(
        title="🌲 Random Forest Feature Importance",
        xaxis_title="Importance", yaxis_title="Feature",
    )
    return fig


def fig_roc_curve(roc_results: dict) -> go.Figure:
    """ROC curve for one or more models. roc_results: {name: (fpr, tpr, auc)}."""
    fig = go.Figure()
    colors = ["#636EFA", "#00CC96", "#EF553B"]
    for i, (name, (fpr, tpr, auc)) in enumerate(roc_results.items()):
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC = {auc:.3f})",
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random Guess (AUC = 0.5)",
        line=dict(color="gray", dash="dash"),
    ))
    fig.update_layout(
        title="📈 ROC Curve",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
    )
    return fig


# ---------------------------------------------------------------------------
# EDA charts (Data Understanding module)
# ---------------------------------------------------------------------------
def fig_histogram(series: pd.Series, title: str, x_label: str) -> go.Figure:
    """Histogram for a numeric column."""
    fig = px.histogram(
        x=series, nbins=40, title=title,
        labels={"x": x_label}, color_discrete_sequence=[ACCENT],
    )
    fig.update_layout(bargap=0.05, yaxis_title="Count")
    return fig


def fig_boxplot(df: pd.DataFrame, y_col: str, x_col: str, title: str) -> go.Figure:
    """Box plot: y_col distribution grouped by x_col (outlier detection)."""
    fig = px.box(
        df, x=x_col, y=y_col, title=title,
        color_discrete_sequence=[ACCENT],
    )
    return fig


def fig_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str,
                 color_col: str = None) -> go.Figure:
    """Scatter plot between two numeric columns."""
    fig = px.scatter(
        df, x=x_col, y=y_col, color=color_col, title=title,
        opacity=0.5, color_discrete_sequence=PALETTE,
    )
    return fig


def fig_countplot(series: pd.Series, title: str, x_label: str) -> go.Figure:
    """Count plot (bar chart of category frequencies)."""
    counts = series.value_counts().reset_index()
    counts.columns = [x_label, "count"]
    fig = px.bar(
        counts, x=x_label, y="count", title=title,
        color_discrete_sequence=[ACCENT],
    )
    return fig


def fig_correlation_heatmap(df: pd.DataFrame, columns: list) -> go.Figure:
    """Correlation heatmap for the given numeric columns."""
    corr = df[columns].corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, title="🔗 Correlation Heatmap",
    )
    return fig
