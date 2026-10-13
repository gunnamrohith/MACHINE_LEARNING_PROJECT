"""
M2 (Part 2): Exploratory Data Analysis Module
==============================================
Objective: Discover business insights from the cleaned, featured dataset.

Provides helper functions that return DataFrames/Series used by
Streamlit pages and the home dashboard.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Sales Analysis
# ---------------------------------------------------------------------------
def total_revenue(df: pd.DataFrame) -> float:
    """Total revenue from all purchase events."""
    return df.loc[df["event_type"] == "purchase", "price"].sum()


def revenue_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue grouped by top-level category, sorted descending."""
    purchases = df[df["event_type"] == "purchase"].copy()
    result = (
        purchases.groupby("category_main")["price"]
        .sum()
        .reset_index()
        .rename(columns={"price": "revenue"})
        .sort_values("revenue", ascending=False)
    )
    return result


def revenue_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue per (year, month) period."""
    purchases = df[df["event_type"] == "purchase"].copy()
    result = (
        purchases.groupby(["year", "month"])["price"]
        .sum()
        .reset_index()
        .rename(columns={"price": "revenue"})
        .sort_values(["year", "month"])
    )
    result["period"] = result["year"].astype(str) + "-" + result["month"].astype(str).str.zfill(2)
    return result


def revenue_trend_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Daily revenue time series."""
    purchases = df[df["event_type"] == "purchase"].copy()
    purchases["date_str"] = purchases["event_time"].dt.date
    result = (
        purchases.groupby("date_str")["price"]
        .sum()
        .reset_index()
        .rename(columns={"date_str": "date", "price": "revenue"})
        .sort_values("date")
    )
    return result


# ---------------------------------------------------------------------------
# Customer Analysis
# ---------------------------------------------------------------------------
def total_customers(df: pd.DataFrame) -> int:
    return df["user_id"].nunique()


def top_customers(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top N customers by total spend."""
    purchases = df[df["event_type"] == "purchase"].copy()
    result = (
        purchases.groupby("user_id")["price"]
        .agg(total_spend="sum", num_purchases="count")
        .reset_index()
        .sort_values("total_spend", ascending=False)
        .head(n)
    )
    return result


def customer_activity_distribution(df: pd.DataFrame) -> pd.Series:
    """Number of events per customer (all event types)."""
    return df.groupby("user_id").size()


# ---------------------------------------------------------------------------
# Product Analysis
# ---------------------------------------------------------------------------
def total_products(df: pd.DataFrame) -> int:
    return df["product_id"].nunique()


def most_viewed_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    views = df[df["event_type"] == "view"].copy()
    result = (
        views.groupby("product_id")
        .size()
        .reset_index(name="view_count")
        .sort_values("view_count", ascending=False)
        .head(n)
    )
    return result


def most_purchased_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    purchases = df[df["event_type"] == "purchase"].copy()
    result = (
        purchases.groupby("product_id")
        .size()
        .reset_index(name="purchase_count")
        .sort_values("purchase_count", ascending=False)
        .head(n)
    )
    return result


def category_popularity(df: pd.DataFrame) -> pd.DataFrame:
    """Event count per top-level category."""
    result = (
        df.groupby("category_main")
        .size()
        .reset_index(name="event_count")
        .sort_values("event_count", ascending=False)
    )
    return result


# ---------------------------------------------------------------------------
# KPI Summary
# ---------------------------------------------------------------------------
def kpi_summary(df: pd.DataFrame) -> dict:
    purchases = df[df["event_type"] == "purchase"]
    return {
        "total_revenue": purchases["price"].sum(),
        "total_customers": df["user_id"].nunique(),
        "total_products": df["product_id"].nunique(),
        "total_purchases": len(purchases),
        "avg_order_value": purchases["price"].mean(),
    }
