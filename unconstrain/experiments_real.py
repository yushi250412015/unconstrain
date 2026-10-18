"""Real-data experiments on FreshRetailNet-50K.

R1  clean-day holdout   -- train on censored history, score the official 7-day
                           holdout on *fully stocked* days only, where sales and
                           demand coincide.  No invented ground truth.
R2  bias law            -- per-series bias against per-series stock-out rate:
                           the number a planner can look up.
R3  closed loop         -- run the real forecasts through an (s, S) policy on
                           real demand and price the bias in lost units.

One fit per method is shared by all three, because refitting is the expensive
part and the three views are just different questions asked of the same
forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .forecast import MAX_LAG, DirectForecaster
from .methods import HARD_ORDER, METHOD_ORDER, reconstruct
from .metrics import inventory_cost
from .simulate import Policy, policy_from_forecast, simulate_inventory

H = 7
GBM = dict(max_iter=50, learning_rate=0.10, random_state=0)

#: the comparison that carries the argument: the same three estimators, with and
#: without the severity weight
CONTRAST = ("naive", "interp", "km", "tobit", "nb", "km_s", "tobit_s", "nb_s")


def fit_predict(panel: dict, holdout: dict, method: str,
                horizons: int = H, **kw) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct history under ``method``, fit, and forecast the holdout."""
    sales, mask = panel["sales"], panel["mask"]
    exog = panel["promo"][:, :, None]
    R = (sales.astype(float) if method == "sales"
         else reconstruct(sales, mask, method, exog,
                          severity=panel.get("severity")))
    origins = np.arange(MAX_LAG + 1, sales.shape[1] - horizons)
    model = DirectForecaster(horizons=horizons, **{**GBM, **kw}).fit(
        R, R, mask, origins, panel["promo"])
    fc = model.predict(R, mask, sales.shape[1] - 1, holdout["promo"])
    return R, fc


def score(fc: np.ndarray, holdout: dict) -> dict:
    """Score on fully-stocked holdout days: sales there are demand."""
    clean = ~holdout["mask"]
    y = holdout["sales"]
    err = fc - y
    denom = y[clean].sum()
    return {
        "wape": float(100.0 * np.abs(err[clean]).sum() / max(denom, 1e-9)),
        "bias": float(100.0 * err[clean].sum() / max(denom, 1e-9)),
        "clean_share": float(clean.mean()),
        "n_points": int(clean.sum()),
    }


def _binned_law(rate, bias, weights, n_bins=5):
    """Volume-weighted mean bias per stock-out-rate bin.

    A per-series regression is hopeless here: each series contributes about
    four clean holdout days, so the per-series ratio is mostly sampling noise
    (R^2 around 0.02).  Binning first turns the same information into five
    well-measured points, which is what the figure should show.
    """
    edges = np.unique(np.quantile(rate, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        return np.array([rate.mean()]), np.array([np.average(bias, weights=weights)])
    idx = np.clip(np.searchsorted(edges, rate, side="right") - 1, 0, edges.size - 2)
    xs, ys = [], []
    for b in range(edges.size - 1):
        sel = idx == b
        if not sel.any():
            continue
        ww = weights[sel]
        xs.append(float(np.average(rate[sel], weights=ww)))
        ys.append(float(np.average(bias[sel], weights=ww)))
    return np.asarray(xs), np.asarray(ys)


def run_real(panel: dict, holdout: dict, methods=CONTRAST,
             policy: Policy | None = None) -> dict:
    """R1 + R2 + R3 from a single forecast per method."""
    pol = policy or Policy(lead_time=3, reorder_days=1.5, cover_days=3.0)
    clean = ~holdout["mask"]
    y = holdout["sales"]
    rate = panel["mask"].mean(axis=1)
    vol = panel["sales"].sum(axis=1)
    w = vol / vol.sum()

    r1, per_series, fc_by_method = [], {}, {}
    abs_sum, sgn_sum = {}, {}
    for m in methods:
        _, fc = fit_predict(panel, holdout, m)
        s = score(fc, holdout)
        s["method"] = m
        r1.append(s)
        err = np.where(clean, fc - y, 0.0)
        num, den = err.sum(axis=1), np.where(clean, y, 0.0).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_series[m] = np.where(den > 0, num / den, np.nan)
        abs_sum[m] = np.where(clean, np.abs(fc - y), 0.0).sum(axis=1)
        sgn_sum[m] = num
        fc_by_method[m] = fc
        print(f"  R1 {m:8s} WAPE {s['wape']:5.1f}  bias {s['bias']:+6.1f}%")

    r1 = pd.DataFrame(r1)[["method", "wape", "bias", "clean_share", "n_points"]]
    # a ratio estimator is weighted by its denominator, not by total volume
    den_all = np.where(clean, y, 0.0).sum(axis=1)
    w = den_all / max(den_all.sum(), 1e-9)
    rows, curves = [], {}
    for m, bias in per_series.items():
        ok = np.isfinite(bias)
        xs, ys = _binned_law(rate[ok], bias[ok], w[ok])
        curves[m] = (xs, ys)
        if xs.size >= 2:
            A = np.stack([np.ones_like(xs), xs], axis=1)
            coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
            pred = A @ coef
            r2 = 1 - float(((ys - pred) ** 2).sum()) / max(float(((ys - ys.mean()) ** 2).sum()), 1e-12)
        else:
            coef, r2 = np.array([float(ys[0]), 0.0]), float("nan")
        rows.append(dict(method=m, intercept=100 * coef[0], slope=100 * coef[1],
                         r2=r2, rate_mean=float(xs.mean()),
                         bias_mean=100 * float(ys.mean())))
    r2 = pd.DataFrame(rows)
    r2.attrs["curves"] = curves
    r2.attrs["per_series"] = per_series
    r2.attrs["rate"] = rate

    r3 = []
    for m, fc in fc_by_method.items():
        daily = np.maximum(fc.mean(axis=1), 0.0)
        r_, S_ = policy_from_forecast(daily, pol)
        sim = simulate_inventory(y, r_, S_, pol)
        cost = inventory_cost(sim["sales"], sim["lost"], sim["inv"],
                              pol.holding_cost, pol.stockout_cost)
        r3.append(dict(method=m, fill_rate=cost["fill_rate"],
                       lost=cost["lost_units"],
                       lost_pct=100 * cost["lost_units"] / max(y.sum(), 1e-9),
                       cost=cost["total"], order_level=float(daily.mean())))
        print(f"  R3 {m:8s} fill {cost['fill_rate']:.3f}  lost {cost['lost_units']:.0f}")
    r3 = pd.DataFrame(r3).sort_values("lost")

    return {"r1": r1, "r2": r2, "r3": r3, "r4": None, "forecasts": fc_by_method,
            "holdout": holdout, "panel": panel,
            "ci": {"abs": abs_sum, "signed": sgn_sum, "den": den_all}}


def bootstrap_ci(res: dict, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Cluster bootstrap over series: a confidence interval for bias and WAPE.

    Resampling *series* (not days) is the right unit: days within a SKU are
    strongly dependent, so resampling them would understate the interval.
    """
    abs_sum, sgn_sum, den = res["ci"]["abs"], res["ci"]["signed"], res["ci"]["den"]
    keep = den > 0
    den = den[keep]
    rng = np.random.default_rng(seed)
    n = den.size
    rows = []
    for m in abs_sum:
        a, s = abs_sum[m][keep], sgn_sum[m][keep]
        wape_boot = np.empty(n_boot)
        bias_boot = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            d = den[idx].sum()
            wape_boot[b] = 100.0 * a[idx].sum() / d
            bias_boot[b] = 100.0 * s[idx].sum() / d
        rows.append(dict(
            method=m,
            wape=100.0 * a.sum() / den.sum(),
            wape_lo=np.percentile(wape_boot, 2.5),
            wape_hi=np.percentile(wape_boot, 97.5),
            bias=100.0 * s.sum() / den.sum(),
            bias_lo=np.percentile(bias_boot, 2.5),
            bias_hi=np.percentile(bias_boot, 97.5),
            n_series=int(n),
        ))
    return pd.DataFrame(rows)


def format_ci(df: pd.DataFrame) -> str:
    lines = [f"R1 with cluster bootstrap 95% CI (n_series={df.n_series.iloc[0]}, "
             f"2000 resamples)"]
    lines.append(f"  {'method':9s} {'WAPE':>6s} {'95% CI':>15s} {'bias':>7s} {'95% CI':>15s}")
    for r in df.itertuples():
        lines.append(f"  {r.method:9s} {r.wape:6.1f} "
                     f"[{r.wape_lo:5.1f},{r.wape_hi:5.1f}] "
                     f"{r.bias:+7.1f} [{r.bias_lo:+6.1f},{r.bias_hi:+6.1f}]")
    return "\n".join(lines)


def _fit_quantile(panel, holdout, R, q):
    origins = np.arange(MAX_LAG + 1, panel["sales"].shape[1] - H)
    model = DirectForecaster(horizons=H, quantile=q, **GBM).fit(
        R, R, panel["mask"], origins, panel["promo"])
    return model.predict(R, panel["mask"], panel["sales"].shape[1] - 1,
                         holdout["promo"])


def run_quantile_study(panel: dict, holdout: dict,
                       methods=("naive", "tobit_s", "nb_s"),
                       quantiles=(0.5, 0.9, 0.95, 0.98),
                       policy: Policy | None = None) -> pd.DataFrame:
    """Bias is not money: order the *quantile* the cost ratio asks for.

    With a shortage worth 50 unit-days of holding, the newsvendor critical
    fractile is 1/(1+0.02) ~ 0.98.  A mean forecast is the wrong target no
    matter how unbiased it is, and this table prices that mistake.
    """
    pol = policy or Policy()
    y = holdout["sales"]
    rows = []
    for m in methods:
        if m == "naive":
            R = panel["sales"].astype(float)
        else:
            R = reconstruct(panel["sales"], panel["mask"], m,
                            panel["promo"][:, :, None],
                            severity=panel.get("severity"))
        for q in quantiles:
            fc = _fit_quantile(panel, holdout, R, q)
            daily = np.maximum(fc.mean(axis=1), 0.0)
            r_, S_ = policy_from_forecast(daily, pol)
            sim = simulate_inventory(y, r_, S_, pol)
            cost = inventory_cost(sim["sales"], sim["lost"], sim["inv"],
                                  pol.holding_cost, pol.stockout_cost)
            rows.append(dict(method=m, q=q, fill_rate=cost["fill_rate"],
                             lost=cost["lost_units"], cost=cost["total"]))
            print(f"  R4 {m:8s} q={q:.2f}  fill {cost['fill_rate']:.3f}  "
                  f"lost {cost['lost_units']:.0f}")
    return pd.DataFrame(rows)


def format_r4(df: pd.DataFrame) -> str:
    lines = ["R4  ordering quantile (newsvendor critical fractile ~ 0.98 "
             "for a 50:1 cost ratio)"]
    head = "  " + f"{'method':9s}" + "".join(f"{q:>9.2f}" for q in sorted(df.q.unique()))
    lines.append(head + "   <- lost units, lower is better")
    for m, sub in df.groupby("method", sort=False):
        sub = sub.sort_values("q")
        lines.append(f"  {m:9s}" + "".join(f"{v:>9.0f}" for v in sub.lost))
    return "\n".join(lines)


def run_foundation(panel: dict, holdout: dict, histories=("raw", "tobit_s"),
                   model: str = "amazon/chronos-bolt-small") -> pd.DataFrame:
    """Zero-shot foundation model, with and without a de-censored history.

    Chronos never sees a stock-out flag, so it can only learn the censored
    level.  Feeding it a reconstructed history is the cheapest possible test of
    whether the de-censoring layer is model-agnostic -- it is, and it is worth
    more than the gap between a tuned GBM and a foundation model.
    """
    import torch
    from chronos import BaseChronosPipeline

    pipe = BaseChronosPipeline.from_pretrained(model, torch_dtype=torch.float32)
    rows = []
    for h in histories:
        hist = (panel["sales"].astype(float) if h == "raw" else
                reconstruct(panel["sales"], panel["mask"], h,
                            panel["promo"][:, :, None],
                            severity=panel.get("severity")))
        ctx = torch.tensor(np.clip(hist, 0.0, None).astype(np.float32))
        _, mean = pipe.predict_quantiles(ctx, prediction_length=H,
                                         quantile_levels=[0.5])
        fc = mean.numpy()
        s = score(fc, holdout)
        s["history"] = h
        rows.append(s)
        print(f"  R5 chronos[{h:8s}] WAPE {s['wape']:5.1f}  bias {s['bias']:+6.1f}%")
    return pd.DataFrame(rows)[["history", "wape", "bias", "n_points"]]


def format_r5(df: pd.DataFrame) -> str:
    lines = ["R5  zero-shot Chronos-Bolt, same clean-day protocol"]
    for r in df.itertuples():
        lines.append(f"  history={r.history:9s} WAPE {r.wape:5.1f}  bias {r.bias:+6.1f}%")
    return "\n".join(lines)


def format_r1(r1: pd.DataFrame, r3: pd.DataFrame) -> str:
    merged = r1.merge(r3, on="method", how="left")
    lines = ["R1/R3  official 7-day holdout (scored on fully-stocked days only)"]
    lines.append(f"  {'method':9s} {'WAPE':>6s} {'bias':>7s} {'fill':>6s} {'lost':>7s}")
    for r in merged.itertuples():
        lines.append(f"  {r.method:9s} {r.wape:6.1f} {r.bias:+7.1f} "
                     f"{r.fill_rate:6.3f} {r.lost:7.0f}")
    return "\n".join(lines)


def format_r2(r2: pd.DataFrame) -> str:
    lines = ["R2  per-series bias vs per-series stock-out rate (volume weighted)"]
    lines.append(f"  {'method':9s} {'slope':>7s} {'intercept':>10s} {'R2':>6s}")
    for r in r2.itertuples():
        lines.append(f"  {r.method:9s} {r.slope:+7.1f} {r.intercept:+10.1f} {r.r2:6.2f}")
    return "\n".join(lines)
