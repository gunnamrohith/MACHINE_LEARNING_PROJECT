"""
M3: Probability Analysis Module
================================
Objective: Understand customer conversion behavior using probability.

Calculations:
- P(View), P(Cart), P(Purchase)       – marginal probabilities
- P(Purchase | View)                   – conditional probability
- P(Purchase | Cart)                   – conditional probability
- View → Cart → Purchase funnel rates
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Event Probabilities
# ---------------------------------------------------------------------------
def event_probabilities(df: pd.DataFrame) -> dict:
    """
    Calculate the marginal probability of each event type.

    P(View)    = count(view)    / total_events
    P(Cart)    = count(cart)    / total_events
    P(Purchase)= count(purchase)/ total_events
    """
    total = len(df)
    counts = df["event_type"].value_counts()
    probs = {}
    for event in ["view", "cart", "purchase"]:
        probs[f"P({event})"] = counts.get(event, 0) / total if total > 0 else 0
    probs["counts"] = counts.to_dict()
    probs["total"] = total
    return probs


# ---------------------------------------------------------------------------
# Conditional Probabilities
# ---------------------------------------------------------------------------
def conditional_probabilities(df: pd.DataFrame) -> dict:
    """
    P(Purchase | View)  – Among sessions that had a view, what fraction also had a purchase?
    P(Purchase | Cart)  – Among sessions that had a cart event, what fraction had a purchase?

    Methodology (session-level):
    A session either resulted in a purchase or it didn't. We group by user_session
    and check which event types occurred, then compute the conditional rates.
    This gives intuitive, business-meaningful numbers.
    """
    session_events = df.groupby("user_session")["event_type"].apply(set)

    sessions_with_view = session_events[session_events.apply(lambda s: "view" in s)]
    sessions_with_cart = session_events[session_events.apply(lambda s: "cart" in s)]
    sessions_with_purchase = session_events[session_events.apply(lambda s: "purchase" in s)]

    n_view = len(sessions_with_view)
    n_cart = len(sessions_with_cart)
    n_purchase = len(sessions_with_purchase)

    # Sessions that had view AND purchase
    view_then_purchase = session_events[
        session_events.apply(lambda s: "view" in s and "purchase" in s)
    ]
    # Sessions that had cart AND purchase
    cart_then_purchase = session_events[
        session_events.apply(lambda s: "cart" in s and "purchase" in s)
    ]

    p_purchase_given_view = len(view_then_purchase) / n_view if n_view > 0 else 0
    p_purchase_given_cart = len(cart_then_purchase) / n_cart if n_cart > 0 else 0

    return {
        "P(Purchase | View)": p_purchase_given_view,
        "P(Purchase | Cart)": p_purchase_given_cart,
        "sessions_with_view": n_view,
        "sessions_with_cart": n_cart,
        "sessions_with_purchase": n_purchase,
        "sessions_view_and_purchase": len(view_then_purchase),
        "sessions_cart_and_purchase": len(cart_then_purchase),
    }


# ---------------------------------------------------------------------------
# Funnel Analysis
# ---------------------------------------------------------------------------
def funnel_analysis(df: pd.DataFrame) -> dict:
    """
    Compute the conversion funnel:
      View → Cart → Purchase

    Rates:
    - View-to-Cart   = count(cart)    / count(view)
    - Cart-to-Purchase = count(purchase) / count(cart)
    - Overall        = count(purchase) / count(view)
    """
    counts = df["event_type"].value_counts()
    n_view = counts.get("view", 1)
    n_cart = counts.get("cart", 0)
    n_purchase = counts.get("purchase", 0)

    return {
        "stages": ["View", "Cart", "Purchase"],
        "counts": [n_view, n_cart, n_purchase],
        "view_to_cart_rate": n_cart / n_view if n_view > 0 else 0,
        "cart_to_purchase_rate": n_purchase / n_cart if n_cart > 0 else 0,
        "overall_conversion_rate": n_purchase / n_view if n_view > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Business Interpretation Helpers
# ---------------------------------------------------------------------------
def interpret_purchase_given_view(p: float) -> str:
    per_100 = round(p * 100, 1)
    return (
        f"Out of every 100 viewed products, approximately {per_100} products "
        f"are eventually purchased. This is your product-level conversion rate."
    )


def interpret_funnel(funnel: dict) -> str:
    v2c = round(funnel["view_to_cart_rate"] * 100, 1)
    c2p = round(funnel["cart_to_purchase_rate"] * 100, 1)
    overall = round(funnel["overall_conversion_rate"] * 100, 1)
    return (
        f"**View → Cart**: {v2c}% of views result in a cart add. "
        f"**Cart → Purchase**: {c2p}% of cart additions result in a purchase. "
        f"**Overall**: Only {overall}% of all product views lead to a purchase — "
        f"optimising the cart-to-purchase step typically gives the highest ROI."
    )
