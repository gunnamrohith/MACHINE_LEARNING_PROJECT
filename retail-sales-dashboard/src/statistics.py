"""
M4: Statistical Inference Module
==================================
Objective: Validate findings using basic statistical methods.

Methods:
1. 95% Confidence Interval for average product price
2. Independent-samples T-Test: weekend vs weekday purchase amounts
"""

import pandas as pd
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Confidence Interval
# ---------------------------------------------------------------------------
def confidence_interval_price(df: pd.DataFrame, confidence: float = 0.95) -> dict:
    """
    Calculate a confidence interval for the mean product price.

    Formula (large-sample normal approximation):
        CI = x̄ ± z * (σ / √n)

    For smaller samples we use the t-distribution via scipy.
    """
    prices = df.loc[df["event_type"] == "purchase", "price"].dropna()
    n = len(prices)
    mean = prices.mean()
    se = stats.sem(prices)   # standard error of the mean
    ci = stats.t.interval(confidence, df=n - 1, loc=mean, scale=se)

    return {
        "mean": mean,
        "std": prices.std(),
        "n": n,
        "lower": ci[0],
        "upper": ci[1],
        "confidence": confidence,
    }


def interpret_ci(ci_result: dict) -> str:
    conf_pct = int(ci_result["confidence"] * 100)
    return (
        f"We are {conf_pct}% confident that the **true average product price** "
        f"lies between **₹{ci_result['lower']:.2f}** and **₹{ci_result['upper']:.2f}**. "
        f"(Sample mean = ₹{ci_result['mean']:.2f}, n = {ci_result['n']:,})"
    )


# ---------------------------------------------------------------------------
# 2. Hypothesis Test – Weekend vs Weekday Purchases
# ---------------------------------------------------------------------------
def ttest_weekend_vs_weekday(df: pd.DataFrame) -> dict:
    """
    H₀: µ_weekend = µ_weekday  (no difference in purchase amount)
    H₁: µ_weekend ≠ µ_weekday  (two-tailed)

    Test: Independent-samples Welch's T-Test
    """
    purchases = df[df["event_type"] == "purchase"].copy()
    weekend = purchases.loc[purchases["is_weekend"], "price"].dropna()
    weekday = purchases.loc[~purchases["is_weekend"], "price"].dropna()

    t_stat, p_value = stats.ttest_ind(weekend, weekday, equal_var=False)

    alpha = 0.05
    conclusion = "reject H₀" if p_value < alpha else "fail to reject H₀"

    return {
        "h0": "Weekend and weekday purchase amounts are equal (µ_weekend = µ_weekday)",
        "h1": "Weekend and weekday purchase amounts are different (µ_weekend ≠ µ_weekday)",
        "weekend_mean": weekend.mean(),
        "weekday_mean": weekday.mean(),
        "weekend_n": len(weekend),
        "weekday_n": len(weekday),
        "t_statistic": t_stat,
        "p_value": p_value,
        "alpha": alpha,
        "conclusion": conclusion,
    }


def interpret_ttest(result: dict) -> str:
    if result["p_value"] < result["alpha"]:
        direction = "higher" if result["weekend_mean"] > result["weekday_mean"] else "lower"
        return (
            f"The T-Test is statistically significant (p = {result['p_value']:.4f} < 0.05). "
            f"We **reject H₀**: weekend purchases (mean = ₹{result['weekend_mean']:.2f}) "
            f"are significantly {direction} than weekday purchases "
            f"(mean = ₹{result['weekday_mean']:.2f}). "
            f"**Business action**: consider weekend-specific promotions."
        )
    else:
        return (
            f"The T-Test is NOT statistically significant (p = {result['p_value']:.4f} ≥ 0.05). "
            f"We **fail to reject H₀**: there is no meaningful difference between "
            f"weekend (mean = ₹{result['weekend_mean']:.2f}) and "
            f"weekday (mean = ₹{result['weekday_mean']:.2f}) purchase amounts. "
            f"**Business action**: use a uniform pricing / promotion strategy."
        )
