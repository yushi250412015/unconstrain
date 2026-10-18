"""Public API.

Three questions, in the order a user actually asks them:

    diagnose(...)        how censored is my data, and what can it support?
    decensor(...)        what was demand, given the censoring I have?
    recommend_quantile() how much should I order, given what it costs?

The design assumption is that data quality is a *ladder*, not a yes/no. Most
teams log sales and nothing else; some log a stock-out flag; few log stock hours.
Each rung buys a measurable amount of recoverable demand, and the point of the
report is to show which rung you are on and what the next one is worth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .identify import fisher_clean, retention
from .methods import reconstruct
from .prep import build_panel, fill_gaps, guess_columns
from .simulate import Policy, policy_from_forecast, simulate_inventory

#: what the literature-measured correction does at each rung of the ladder.
#: (bias of the corrected forecast on clean holdout days, %)
RUNG_BIAS = {
    "R0": None,           # no censoring information at all
    "R1": (9.4, 28.0),    # binary flag -> the textbook correction over-shoots
    "R2": (-0.3, 1.5),    # severity -> lands near zero
    "R3": (-0.3, 1.5),    # own hourly profile -> same, but self-calibrated
}


@dataclass
class Report:
    """What a censoring audit found."""

    n_series: int
    n_days: int
    censoring_rate: float
    mean_severity: float | None
    readiness: str
    information_retention: float
    effective_days: float
    cr_floor_pct: float
    volume_at_risk: float | None = None
    coverage: float = 1.0
    data_issues: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        sev = "n/a" if self.mean_severity is None else f"{self.mean_severity:.0%}"
        lines = [
            "censoring audit",
            f"  series                 {self.n_series}",
            f"  days                   {self.n_days}",
            f"  coverage               {self.coverage:.1%} of series-days observed",
            f"  censored days          {self.censoring_rate:.1%} of observed",
            f"  mean severity          {sev}",
            f"  readiness              {self.readiness}",
            f"  information retained   {self.information_retention:.0%}"
            f"  ({self.effective_days:.0f} effective days)",
            f"  level error floor      {self.cr_floor_pct:.1f}% (any unbiased estimate)",
        ]
        if self.volume_at_risk is not None:
            lines.append(f"  volume at risk         {self.volume_at_risk:.1%} of sales")
        if self.data_issues:
            lines.append("  data issues:")
            lines += [f"    - {i}" for i in self.data_issues]
        if self.missing:
            lines.append("  missing:")
            lines += [f"    - {m}" for m in self.missing]
        if self.actions:
            lines.append("  next:")
            lines += [f"    {i+1}. {a}" for i, a in enumerate(self.actions)]
        return "\n".join(lines)


def readiness_level(*, has_severity: bool, has_flag: bool, hourly: bool) -> str:
    """Which rung of the ladder the available columns support."""
    if hourly:
        return "R3"
    if has_severity:
        return "R2"
    if has_flag:
        return "R1"
    return "R0"


def _columns(df: pd.DataFrame, sales, series, date, stockout, stock_hours, severity):
    """Use the caller's names when they exist, otherwise look like a human."""
    guess = guess_columns(df, sales=sales, series=series, date=date,
                          stockout=stockout, stock_hours=stock_hours,
                          severity=severity)
    if sales in df.columns and series in df.columns and date in df.columns:
        guess.update(sales=sales, series=series, date=date)
    if guess["sales"] is None or guess["series"] is None or guess["date"] is None:
        raise ValueError(
            "could not identify the sales/series/date columns; pass them "
            f"explicitly. Columns seen: {list(df.columns)[:12]}")
    return guess


def diagnose(df: pd.DataFrame, *, sales="sales", series="series", date="date",
             stockout=None, severity=None, stock_hours=None, open_hours=16,
             hourly=False) -> Report:
    """Audit a sales table for censoring and say what it can and cannot support.

    Column names are arguments so that the function works on a raw export; if
    they do not exist, the columns are guessed from their names.

    ======  =============================  =================================
    rung    you log                        consequence
    ======  =============================  =================================
    R0      sales                          nothing recoverable, bias unknown
    R1      + a stock-out flag             correction over-shoots 9-28%
    R2      + stock hours or severity      correction lands near 0%
    R3      + hourly sales                 same, and self-calibrated
    ======  =============================  =================================
    """
    cols = _columns(df, sales, series, date, stockout, stock_hours, severity)
    panel = build_panel(df, sales=cols["sales"], series=cols["series"],
                        date=cols["date"], stockout=cols["stockout"],
                        stock_hours=cols["stock_hours"],
                        severity=cols["severity"], open_hours=open_hours)

    obs, mask, sev = panel.observed, panel.mask, panel.severity
    n_obs = int(obs.sum())
    rate = float(mask.sum() / max(n_obs, 1))
    mgr = float(sev[mask].mean()) if (sev is not None and mask.any()) else None

    # per-series level from clean observations
    clean = obs & ~mask
    with np.errstate(invalid="ignore"):
        level = np.nanmean(np.where(clean, panel.values, np.nan), axis=1)
    fallback = float(np.nanmedian(level)) if np.isfinite(level).any() else 1.0
    level = np.where(np.isfinite(level) & (level > 0.05), level, max(fallback, 0.3))

    # information accounting: a clean observed day keeps rho=1, a censored day
    # keeps rho(s), an unobserved day keeps nothing
    if sev is not None:
        rhos = np.vectorize(lambda s: retention(1.0, 60.0, float(s)))(sev)
    else:
        rhos = np.full_like(panel.values, 0.34)
    rho = np.where(obs, np.where(mask, rhos, 1.0), 0.0)
    n_eff = float(np.median(rho.sum(axis=1)))

    i1 = float(np.median([fisher_clean(m, 60.0) for m in np.clip(level, 0.3, 40)]))
    cr = float(1.0 / np.sqrt(max(n_eff * i1, 1e-12)) / max(float(np.median(level)), 1e-9))

    lvl = readiness_level(has_severity=bool(cols["severity"] or cols["stock_hours"]),
                          has_flag=bool(cols["stockout"] or cols["stock_hours"]),
                          hourly=hourly)

    data_issues = [f"{i['kind']}: {i['count']} - {i['detail']}" for i in panel.issues]
    missing, actions = [], []
    if lvl == "R0":
        missing.append("no stock-out information: censoring is unidentifiable")
        actions.append("log whether each SKU was out of stock at end of day "
                       "(one boolean column, usually already in the WMS)")
    if lvl == "R1":
        missing.append("stock-out hours or severity: a binary flag over-corrects")
        actions.append("replace the flag with the number of unavailable selling "
                       "hours; that moves the correction from +9..28% to about 0")
        actions.append("until then, use strength=0.4 on decensor() to avoid "
                       "over-correcting")
    if lvl in ("R2", "R3"):
        actions.append("de-censor the history with decensor(), then order the "
                       "quantile from recommend_quantile() - not the mean")
    if hourly:
        actions.append("estimate the hourly profile from your own clean days "
                       "(the R3 variant) and re-run")

    return Report(
        n_series=panel.n_series, n_days=panel.n_days,
        censoring_rate=rate, mean_severity=mgr, readiness=lvl,
        information_retention=float(n_eff / max(panel.n_days, 1)),
        effective_days=n_eff, cr_floor_pct=100.0 * cr,
        volume_at_risk=(float(mgr * rate) if mgr is not None else None),
        coverage=float(n_obs / max(panel.values.size, 1)),
        data_issues=data_issues, missing=missing, actions=actions,
    )


def decensor(df: pd.DataFrame, *, sales="sales", series="series", date="date",
             stockout=None, severity=None, stock_hours=None, open_hours=16,
             method="auto", strength=1.0, complete: bool = False) -> pd.DataFrame:
    """Return the table with a ``demand_hat`` column.

    ``method`` is one of ``tobit``, ``nb``, ``km``, ``interp``, ``mean``; the
    first three are censoring-aware and switch to their severity-weighted
    variant automatically when severity is available. ``strength`` blends
    between raw sales (0) and the full correction (1) — the over-correction
    guard for the R1 case.

    Rows for series-days that were never observed come back as NaN rather than
    a fabricated number.
    """
    cols = _columns(df, sales, series, date, stockout, stock_hours, severity)
    panel = build_panel(df, sales=cols["sales"], series=cols["series"],
                        date=cols["date"], stockout=cols["stockout"],
                        stock_hours=cols["stock_hours"],
                        severity=cols["severity"], open_hours=open_hours)

    obs, mask, sev = panel.observed, panel.mask, panel.severity
    base = fill_gaps(panel.values, obs)          # model input only
    soft = {"tobit": "tobit_s", "nb": "nb_s", "km": "km_s"}
    if method == "auto":
        # the ladder decides: severity if we have it, a damped hard correction
        # if we only have a flag, interpolation if we have nothing
        if sev is not None and np.any(sev > 0):
            use = "tobit_s"  # keep the caller's strength
        elif mask.any():
            use, strength = "tobit", min(strength, 0.4)
        else:
            use = "interp"
    else:
        use = (soft.get(method, method)
               if (sev is not None and np.any(sev > 0)) else method)
    rec = reconstruct(base, mask, use, severity=sev)
    if strength != 1.0:
        rec = base + float(strength) * (rec - base)
    rec = np.where(obs, rec, np.nan)

    flat = pd.DataFrame({
        "series": np.repeat(np.asarray(panel.series), panel.n_days),
        "date": np.tile(pd.to_datetime(panel.dates), panel.n_series),
        "demand_hat": np.asarray(rec, dtype=float).reshape(-1),
    }).rename(columns={"series": series, "date": date})
    d = df.copy()
    d[date] = pd.to_datetime(d[date], errors="coerce")
    if complete:
        grid = pd.MultiIndex.from_product(
            [np.asarray(panel.series), pd.to_datetime(panel.dates)],
            names=[series, date]).to_frame(index=False)
        return grid.merge(flat, on=[series, date], how="left")
    return d.merge(flat, on=[series, date], how="left")


def recommend_quantile(cost_shortage: float, cost_holding: float,
                       *, service_floor: float | None = None) -> dict:
    """Newsvendor critical fractile and the cost of ordering the mean instead.

    The ratio is the only input: ``q* = Cu / (Cu + Co)``. A shortage worth 50
    unit-days of holding implies ordering the 98th percentile, not the mean.
    """
    q = cost_shortage / (cost_shortage + cost_holding)
    if service_floor is not None:
        q = max(q, float(service_floor))
    from scipy.stats import norm
    z = float(norm.ppf(q))
    loss = float(norm.pdf(z) - z * (1.0 - norm.cdf(z)))   # unit normal loss
    return {
        "quantile": round(float(q), 4),
        "z": round(z, 3),
        "expected_cost_at_q": round(loss * cost_shortage, 4),
        "note": "order this quantile of the predictive distribution, not the mean",
    }


def cost_of_policy(df: pd.DataFrame, demand_col: str = "demand_hat", *,
                   series: str = "series", date: str = "date",
                   lead_time: int = 3, cover_days: float = 3.0,
                   cost_shortage: float = 1.0, cost_holding: float = 0.02) -> dict:
    """Run an (s,S) policy over a corrected demand panel and return fill/cost."""
    pol = Policy(lead_time=lead_time, cover_days=cover_days,
                 stockout_cost=cost_shortage, holding_cost=cost_holding)
    s_col = series if series in df.columns else df.columns[0]
    d_col = date if date in df.columns else df.columns[1]
    piv = df.pivot_table(index=s_col, columns=d_col, values=demand_col,
                         aggfunc="first")
    dem = np.nan_to_num(piv.to_numpy(dtype=float))
    daily = dem.mean(axis=1)
    r, S = policy_from_forecast(daily, pol)
    sim = simulate_inventory(dem, r, S, pol)
    lost = float(sim["lost"].sum())
    sold = float(sim["sales"].sum())
    return {
        "fill_rate": round(sold / max(sold + lost, 1e-9), 4),
        "lost_units": round(lost, 1),
        "cost": round(cost_shortage * lost + cost_holding * float(sim["inv"].sum()), 2),
    }
