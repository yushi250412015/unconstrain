"""Synthetic world: latent demand, censoring, and a (s, S) inventory policy.

Only ``demand`` is hidden from the forecaster.  ``observed`` (sales) and
``mask`` (whether the day was stock-constrained) are what a real system sees.
Everything else here exists so that the *error* of every method can be
measured against ground truth instead of against sales.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
@dataclass
class DemandConfig:
    """Parameters of the latent demand process."""

    n_series: int = 40
    n_days: int = 420
    #: fraction of SKUs that are slow movers (zero-inflated, low volume)
    intermittent_frac: float = 0.35
    promo_prob: float = 0.02
    #: Poisson-Gamma overdispersion (NB2 size).  smaller = noisier counts
    overdispersion: float = 60.0
    seed: int = 0
    #: optional multiplicative level shift applied from this day on
    drift_day: int | None = None
    drift_mult: float = 1.0


@dataclass
class Policy:
    """A simple daily-review (s, S) policy sized from a demand forecast.

    Costs are expressed per unit: holding is per unit-day, shortage is per lost
    unit.  A shortage is worth 50 unit-days of holding, which is the ordinary
    retail asymmetry that makes accuracy a poor proxy for profit.
    """

    lead_time: int = 3
    #: reorder point, in days of forecast demand
    reorder_days: float = 1.5
    #: order-up-to level, in days of forecast demand
    cover_days: float = 3.0
    holding_cost: float = 0.02
    stockout_cost: float = 1.0


# --------------------------------------------------------------------------
# demand
# --------------------------------------------------------------------------
def simulate_demand(cfg: DemandConfig = DemandConfig()) -> dict:
    """Draw a panel of demand series with trend, seasonality and promotions.

    Returns a dict with ``demand`` (int, n x T), plus ``mu``, ``promo`` and
    ``intermittent`` for diagnostics.
    """
    rng = np.random.default_rng(cfg.seed)
    n, T = cfg.n_series, int(cfg.n_days)
    t = np.arange(T, dtype=float)

    base = rng.lognormal(mean=1.10, sigma=0.85, size=(n, 1))
    trend = rng.normal(0.0, 0.0009, size=(n, 1))
    amp_w = rng.uniform(0.08, 0.40, size=(n, 1))
    phase_w = rng.integers(0, 7, size=(n, 1)).astype(float)
    amp_y = rng.uniform(0.05, 0.25, size=(n, 1))
    phase_y = rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))
    intermittent = rng.random((n, 1)) < cfg.intermittent_frac
    zero_infl = np.where(intermittent, rng.uniform(0.55, 0.85, size=(n, 1)), 0.0)

    weekly = 1.0 + amp_w * np.sin(2.0 * np.pi * (t[None, :] + phase_w) / 7.0)
    annual = 1.0 + amp_y * np.sin(2.0 * np.pi * t[None, :] / 365.0 + phase_y)
    level = base * (1.0 + trend * t[None, :]) * weekly * annual

    promo = (rng.random((n, T)) < cfg.promo_prob).astype(float)
    lift = np.where(promo > 0.0, rng.uniform(1.6, 2.8, size=(n, T)), 1.0)
    mu = np.maximum(level * lift, 1e-3)

    if cfg.drift_day is not None:
        mu = mu.copy()
        mu[:, int(cfg.drift_day):] *= cfg.drift_mult

    # negative binomial via a Poisson-Gamma mixture
    r = np.where(intermittent, 1.5, cfg.overdispersion)
    lam = rng.gamma(shape=r, scale=mu / r)
    demand = rng.poisson(lam)

    # zero inflation for slow movers; the returned mu must be the mean of the
    # process we actually sample from, not the pre-inflation latent intensity
    mu_eff = mu.copy()
    if intermittent.any():
        zeros = rng.random((n, T)) < zero_infl
        demand = np.where(zeros & intermittent, 0, demand)
        mu_eff = np.where(intermittent, mu * (1.0 - zero_infl), mu)

    return {
        "demand": demand.astype(np.int64),
        "mu": mu_eff,
        "promo": promo,
        "intermittent": intermittent.ravel(),
    }


# --------------------------------------------------------------------------
# censoring
# --------------------------------------------------------------------------
def _threshold_for_rate(series: np.ndarray, rate: float) -> float:
    """Largest cap whose induced censoring rate is closest to ``rate``.

    Cancelling at an integer is coarse: with tied counts, "strictly above q"
    can land far from the requested rate.  We search the observed support
    instead of trusting a quantile interpolation.
    """
    candidates = np.unique(series)[:-1]  # never cap at the series maximum
    best, best_gap = np.inf, np.inf
    for c in candidates:
        gap = abs(float(np.mean(series > c)) - rate)
        if gap < best_gap:
            best, best_gap = float(c), gap
    return best


def apply_censoring(demand: np.ndarray, rate: float, rng=None) -> dict:
    """Exogenous censoring: cap each series at its own empirical quantile.

    Used for controlled experiments, because it lets the *censoring rate* be a
    knobs rather than a consequence.  ``rate`` is the target share of days that
    are stock-constrained.
    """
    demand = np.asarray(demand)
    n, T = demand.shape
    thr = np.full((n, 1), np.inf)
    for i in range(n):
        if rate <= 0.0 or demand[i].max() == 0:
            continue
        thr[i, 0] = _threshold_for_rate(demand[i], rate)

    mask = demand > thr
    observed = np.where(mask, np.minimum(demand, thr), demand).astype(float)
    return {
        "observed": observed,
        "mask": mask,
        "threshold": thr,
        "rate": float(mask.mean()),
    }


# --------------------------------------------------------------------------
# inventory
# --------------------------------------------------------------------------
def simulate_inventory(
    demand: np.ndarray,
    reorder_point: np.ndarray,
    order_up_to: np.ndarray,
    policy: Policy = Policy(),
    start_inv: np.ndarray | None = None,
    pipeline: np.ndarray | None = None,
) -> dict:
    """Run a daily-review (s, S) policy against *true* demand.

    On each day: receive the arriving order, satisfy what demand you can, then
    review and order if on-hand is at or below the reorder point.

    ``start_inv`` / ``pipeline`` let one window continue where the last ended,
    which is what makes the closed-loop experiment a loop rather than a
    sequence of independent weeks.

    Returns per-day ``sales`` (what you observe), ``mask`` (stock-constrained
    days), ``lost`` (unobserved demand), ``inv`` (end-of-day on hand), plus the
    carried ``inv_end`` and ``pipeline_end``.
    """
    demand = np.asarray(demand, dtype=float)
    n, T = demand.shape
    L = int(policy.lead_time)
    reorder_point = np.asarray(reorder_point, dtype=float).reshape(-1)
    order_up_to = np.asarray(order_up_to, dtype=float).reshape(-1)

    if start_inv is None:
        start_inv = order_up_to.copy()
    inv = np.asarray(start_inv, dtype=float).reshape(-1).copy()

    if pipeline is not None:
        pipe = np.array(pipeline, dtype=float)
    else:
        pipe = np.zeros((n, max(L, 1)), dtype=float)
        if L > 0:
            pipe[:, :] = (order_up_to / max(L, 1)).reshape(-1, 1)
    pipeline_ = pipe

    sales = np.zeros((n, T))
    mask = np.zeros((n, T), dtype=bool)
    lost = np.zeros((n, T))
    inv_path = np.zeros((n, T))

    for tt in range(T):
        if L > 0:
            inv = inv + pipeline_[:, 0]
            pipeline_ = np.roll(pipeline_, -1, axis=1)
            pipeline_[:, -1] = 0.0

        avail = np.maximum(inv, 0.0)
        d = demand[:, tt]
        sold = np.minimum(d, avail)
        lost[:, tt] = d - sold
        mask[:, tt] = d > avail
        inv = avail - sold
        sales[:, tt] = sold
        inv_path[:, tt] = inv

        below = inv <= reorder_point
        if below.any():
            idx = np.nonzero(below)[0]
            order = np.maximum(order_up_to[idx] - inv[idx], 0.0)
            if L == 0:
                inv[idx] += order
            else:
                pipeline_[idx, -1] += order

    return {
        "sales": sales,
        "mask": mask,
        "lost": lost,
        "inv": inv_path,
        "inv_end": inv,
        "pipeline_end": pipeline_,
    }


def policy_from_forecast(
    daily_forecast: np.ndarray, policy: Policy = Policy()
) -> tuple[np.ndarray, np.ndarray]:
    """Translate a daily demand forecast into (reorder point, order-up-to).

    This is the coupling that turns a forecast error into an inventory error:
    a forecast that is 12% low shrinks every buffer by 12%.
    """
    daily = np.maximum(np.asarray(daily_forecast, dtype=float).reshape(-1), 0.0)
    return policy.reorder_days * daily, policy.cover_days * daily
