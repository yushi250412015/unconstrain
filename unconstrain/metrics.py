"""Metrics.

The central trick of this project is the *dual target*: every forecast is
scored twice, once against the sales you actually observed and once against
the demand that actually occurred.  A method can win the first and lose the
second, and that gap is the whole story.
"""

from __future__ import annotations

import numpy as np


def bias_pct(pred: np.ndarray, truth: np.ndarray) -> float:
    """Signed aggregate error in percent.  Negative = under-forecast."""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    denom = truth.sum()
    if denom <= 0:
        return float("nan")
    return float(100.0 * (pred.sum() - denom) / denom)


def wape(pred: np.ndarray, truth: np.ndarray) -> float:
    """Weighted absolute percentage error, weighted by the truth.

    Preferred over MAPE for intermittent series: zeros do not explode it.
    """
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    denom = np.abs(truth).sum()
    if denom <= 0:
        return float("nan")
    return float(100.0 * np.abs(pred - truth).sum() / denom)


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.abs(np.asarray(pred, float) - np.asarray(truth, float)).mean())


def inventory_cost(
    sales: np.ndarray,
    lost: np.ndarray,
    inv: np.ndarray,
    holding_cost: float = 1.0,
    stockout_cost: float = 8.0,
) -> dict:
    """Decompose the cost of running a policy.

    Holding is charged on end-of-day stock; stock-outs are charged per lost
    unit, which is the asymmetric part that makes accuracy a bad proxy.
    """
    holding = float(np.asarray(inv, float).sum()) * holding_cost
    shortage = float(np.asarray(lost, float).sum()) * stockout_cost
    return {
        "holding": holding,
        "shortage": shortage,
        "total": holding + shortage,
        "lost_units": float(np.asarray(lost, float).sum()),
        "fill_rate": float(
            np.asarray(sales, float).sum()
            / max(np.asarray(sales, float).sum() + np.asarray(lost, float).sum(), 1e-9)
        ),
    }


def coverage(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> float:
    """Empirical coverage of a nominal interval."""
    lo, hi, y = (np.asarray(a, float) for a in (lo, hi, y))
    return float(np.mean((y >= lo) & (y <= hi)))


def smape(pred: np.ndarray, truth: np.ndarray) -> float:
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    denom = np.abs(pred) + np.abs(truth)
    out = np.where(denom > 0, 2.0 * np.abs(pred - truth) / np.maximum(denom, 1e-9), 0.0)
    return float(100.0 * out.mean())
