"""The three experiments.

E1  bias ladder      -- how far each reconstruction drifts as availability falls
E2  closed loop      -- the bias feeds the policy, which feeds the bias
E3  drift blind spot -- censoring hides a level shift from the monitors

All three use the same forecaster, so differences come from the data the model
is shown, never from model capacity.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .drift import DETECTORS
from .forecast import MAX_LAG, DirectForecaster
from .methods import HARD_ORDER, METHOD_ORDER, inflation, reconstruct
from .metrics import bias_pct, inventory_cost, wape
from .simulate import (DemandConfig, Policy, policy_from_forecast,
                       simulate_demand, simulate_inventory)

H = 7
GBM = dict(max_iter=60, learning_rate=0.10, random_state=0)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def censor_by_policy(dem: np.ndarray, cover_days: float,
                     policy: Policy | None = None) -> dict:
    """Generate censoring from an (s, S) policy sized on *true* demand.

    Sizing on the truth is deliberate: this experiment studies what censoring
    does to a forecaster, so the stock-out mechanism must be independent of any
    forecasting method under test.
    """
    pol = Policy(**(policy.__dict__ if policy else Policy().__dict__))
    pol.cover_days = float(cover_days)
    r_, S_ = policy_from_forecast(dem[:, :28].mean(axis=1), pol)
    return simulate_inventory(dem, r_, S_, pol)


def rolling_eval(R_feat, R_target, mask, promo, train_end: int, test_end: int,
                 horizons: int = H, **kw):
    """Fit once on the training window, then roll origin forward in blocks."""
    origins = np.arange(MAX_LAG + 1, train_end - horizons)
    model = DirectForecaster(horizons=horizons, **kw).fit(
        R_feat, R_target, mask, origins, promo)
    preds, idxs = [], []
    o = train_end - 1
    while o + horizons < test_end:
        preds.append(model.predict(R_feat, mask, o,
                                   promo[:, o + 1:o + 1 + horizons]))
        idxs.append(np.arange(o + 1, o + 1 + horizons))
        o += horizons
    return np.concatenate(preds, axis=1), np.concatenate(idxs)


# --------------------------------------------------------------------------
# E1: the bias ladder
# --------------------------------------------------------------------------
def run_e1(cfg: DemandConfig | None = None, covers=(3.5, 5.0, 7.0, 11.0),
           train_end: int = 300, test_end: int = 370,
           methods: tuple[str, ...] = tuple(HARD_ORDER),
           seeds: tuple[int, ...] = (0, 1, 2)) -> pd.DataFrame:
    """Bias against true demand, swept over availability.

    Averaged over several seeds: a single draw of 40 SKUs carries enough
    promotion-driven noise to reorder the curves, and the point of the figure is
    the *gap between methods*, not one seed's luck.
    """
    cfg = cfg or DemandConfig()
    rows = []
    for seed in seeds:
        world = simulate_demand(replace(cfg, seed=seed))
        dem, promo = world["demand"], world["promo"]
        for cover in covers:
            sim = censor_by_policy(dem, cover)
            obs, msk = sim["sales"], sim["mask"]
            for meth in ("oracle",) + methods:
                if meth == "oracle":
                    R = dem.astype(float)
                else:
                    R = reconstruct(obs, msk, meth, exog=promo[:, :, None])
                P, idx = rolling_eval(R, R, msk, promo, train_end, test_end, **GBM)
                truth, seen = dem[:, idx], obs[:, idx]
                rows.append(dict(
                    seed=seed, cover=cover, rate=float(msk.mean()), method=meth,
                    bias=bias_pct(P, truth),
                    wape_true=wape(P, truth),
                    wape_obs=wape(P, seen),
                    inflation=inflation(obs, msk, R),
                ))
        print(f"  E1 seed={seed} done")
    raw = pd.DataFrame(rows)
    agg = (raw.groupby(["cover", "method"], as_index=False)
              [["rate", "bias", "wape_true", "wape_obs", "inflation"]].mean())
    return agg.sort_values(["cover", "method"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# E2: the closed loop
# --------------------------------------------------------------------------
def run_e2(cfg: DemandConfig | None = None,
           methods: tuple[str, ...] = ("naive", "interp", "km", "tobit"),
           n_iter: int = 14, train_len: int = 210,
           policy: Policy | None = None,
           seeds: tuple[int, ...] = (0, 1)) -> pd.DataFrame:
    """Forecast -> order -> stock out -> observe less -> forecast again.

    The policy is resized every period from the method's own forecast.  This is
    the part that is not visible in any backtest: a low forecast does not only
    score badly, it *changes the data you get next*.
    """
    cfg = cfg or DemandConfig()
    pol = policy or Policy()
    frames = []
    for seed in seeds:
        frames.append(_closed_loop_one(simulate_demand(replace(cfg, seed=seed)),
                                       methods, n_iter, train_len, pol, seed))
        print(f"  E2 seed={seed} done")
    df = pd.concat(frames, ignore_index=True)
    cols = ["stockout", "lost", "cost", "cum_lost", "cum_cost", "fill_rate",
            "forecast_level", "true_level", "inventory"]
    return (df.groupby(["method", "iteration"], as_index=False)[cols].mean()
              .sort_values(["method", "iteration"]).reset_index(drop=True))


def _closed_loop_one(world, methods, n_iter, train_len, pol, seed=0) -> pd.DataFrame:
    dem, promo = world["demand"], world["promo"]
    n, T = dem.shape
    records = []

    for meth in methods + ("oracle",):
        hist_obs = dem[:, :train_len].astype(float).copy()
        hist_mask = np.zeros((n, train_len), dtype=bool)
        level0 = np.maximum(hist_obs.mean(axis=1), 0.1)
        inv = pol.cover_days * level0
        pipe = np.tile(level0.reshape(-1, 1), (1, max(pol.lead_time, 1)))

        cum_lost = cum_cost = 0.0
        for it in range(n_iter):
            a = train_len + it * H
            b = min(a + H, T)
            if b - a < H:
                break
            if meth == "oracle":
                R = dem[:, :a].astype(float)
            else:
                R = reconstruct(hist_obs, hist_mask, meth,
                                exog=promo[:, :a, None])
            model = DirectForecaster(horizons=H, **GBM).fit(
                R, R, hist_mask, np.arange(MAX_LAG + 1, a - H), promo[:, :a])
            fc = model.predict(R, hist_mask, a - 1, promo[:, a:b])
            daily = np.maximum(fc.mean(axis=1), 0.0)
            r_, S_ = policy_from_forecast(daily, pol)

            sim = simulate_inventory(dem[:, a:b], r_, S_, pol,
                                     start_inv=inv, pipeline=pipe)
            inv, pipe = sim["inv_end"], sim["pipeline_end"]
            hist_obs = np.concatenate([hist_obs, sim["sales"]], axis=1)
            hist_mask = np.concatenate([hist_mask, sim["mask"]], axis=1)

            cost = inventory_cost(sim["sales"], sim["lost"], sim["inv"],
                                  pol.holding_cost, pol.stockout_cost)
            cum_lost += cost["lost_units"]
            cum_cost += cost["total"]
            records.append(dict(
                method=meth, iteration=it,
                stockout=float(sim["mask"].mean()),
                lost=cost["lost_units"], cost=cost["total"],
                cum_lost=cum_lost, cum_cost=cum_cost,
                fill_rate=cost["fill_rate"],
                forecast_level=float(daily.mean()),
                true_level=float(dem[:, a:b].mean()),
                inventory=float(np.mean(np.maximum(inv, 0.0))),
            ))
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# E3: the drift blind spot
# --------------------------------------------------------------------------
def _streams(obs, dem, mask, promo, train_end, horizons=1):
    """Three residual streams from one frozen model.

    Each step de-censors the history afresh, forecasts the next day, and scores
    the day that has just closed.  The de-censored stream is therefore
    *available at the same time* as the sales stream -- no look-ahead.
    """
    R0 = reconstruct(obs[:, :train_end], mask[:, :train_end], "km",
                     exog=promo[:, :train_end, None])
    model = DirectForecaster(horizons=horizons, **GBM).fit(
        R0, R0, mask[:, :train_end],
        np.arange(MAX_LAG + 1, train_end - horizons), promo[:, :train_end])

    out = {"demand": [], "sales": [], "de-censored": [], "censor-rate": []}
    prev = None
    for t in range(train_end - 1, dem.shape[1] - horizons):
        R = reconstruct(obs[:, :t + 1], mask[:, :t + 1], "km",
                        exog=promo[:, :t + 1, None])
        if prev is not None:
            j = slice(t, t + 1)
            predicted = float(np.sum(prev))
            out["demand"].append(predicted / max(dem[:, j].sum(), 1e-9) - 1.0)
            out["sales"].append(predicted / max(obs[:, j].sum(), 1e-9) - 1.0)
            out["de-censored"].append(predicted / max(R[:, j].sum(), 1e-9) - 1.0)
            # the mask is not a residual -- it is the constraint itself, and it
            # reacts the day availability stops covering demand
            out["censor-rate"].append(float(mask[:, t].mean()))
        prev = model.predict(R, mask[:, :t + 1], t,
                             promo[:, t + 1:t + 1 + horizons])
    return {k: np.asarray(v, float) for k, v in out.items()}


def run_e3(cfg: DemandConfig | None = None, drift_day: int = 380,
           drift_mult: float = 1.35, covers=(3.0, 2.0, 1.0),
           train_end: int = 300) -> tuple[pd.DataFrame, dict]:
    """Is a level shift still visible once inventory caps the sales?

    Availability is swept because the answer is not binary: as the stock-out
    rate rises, the sales stream does not merely get noisier, it changes sign.
    """
    cfg = replace(cfg or DemandConfig(), n_days=560, drift_day=drift_day,
                  drift_mult=drift_mult)
    world = simulate_demand(cfg)
    dem, promo = world["demand"], world["promo"]

    worlds = {"no censoring": (dem.astype(float), np.zeros_like(dem, bool))}
    for cover in covers:
        sim = censor_by_policy(dem, cover)
        worlds[f"censored (cover={cover:g})"] = (sim["sales"], sim["mask"])

    rows, kept = [], {}
    drift_i = drift_day - train_end
    for wname, (obs, msk) in worlds.items():
        streams = _streams(obs, dem, msk, promo, train_end)
        kept[wname] = {"streams": streams, "rate": float(msk.mean())}
        for sname, series in streams.items():
            if sname == "censor-rate" and msk.mean() <= 0:
                continue  # nothing is ever constrained: the stream is constant
            scale = float(np.std(series[:max(drift_i, 1)])) + 1e-9
            z = series / scale
            for dname, factory in DETECTORS.items():
                det = factory()
                alarm_at, false_alarms = None, 0
                for i, v in enumerate(z):
                    if det.update(v):
                        if i < drift_i:
                            false_alarms += 1
                        elif alarm_at is None:
                            alarm_at = i
                delay = (alarm_at - drift_i) if alarm_at is not None else np.nan
                rows.append(dict(world=wname, stream=sname, detector=dname,
                                 delay=delay, false_alarms=false_alarms,
                                 censoring=float(msk.mean()),
                                 post_drift_bias=float(np.mean(series[drift_i:]))))
        print(f"  E3 {wname} done")
    return pd.DataFrame(rows), kept


# --------------------------------------------------------------------------
# summary printing
# --------------------------------------------------------------------------
def format_e1(df: pd.DataFrame) -> str:
    order = ["oracle"] + [m for m in HARD_ORDER if m in set(df.method)]
    lines = []
    header = f"{'method':10s}" + "".join(f"{r:>9.3f}" for r in sorted(df.rate.unique()))
    lines.append("bias vs TRUE demand (%), by realised censoring rate")
    lines.append(header)
    for m in order:
        sub = df[df.method == m].sort_values("rate")
        lines.append(f"{m:10s}" + "".join(f"{v:>9.1f}" for v in sub.bias))
    return "\n".join(lines)


def format_e2(df: pd.DataFrame) -> str:
    lines = ["closed loop: one policy, one demand stream, different reconstructions"]
    lines.append(f"  {'method':8s} {'stockout 1st':>13s} {'last':>7s} "
                 f"{'fill':>7s} {'lost':>7s} {'cost':>8s}")
    for m, sub in df.groupby("method", sort=False):
        sub = sub.sort_values("iteration")
        lines.append(f"  {m:8s} {sub.stockout.iloc[0]:13.1%} {sub.stockout.iloc[-1]:7.1%} "
                     f"{sub.fill_rate.iloc[-1]:7.3f} {sub.cum_lost.iloc[-1]:7.0f} "
                     f"{sub.cum_cost.iloc[-1]:8.0f}")
    return "\n".join(lines)


def format_e3(df: pd.DataFrame) -> str:
    lines = ["level shift of +35% injected at day 380; residual after the shift "
             "and days until an alarm"]
    for (w, s), sub in df.groupby(["world", "stream"], sort=False):
        vals = ", ".join(f"{r.detector}={r.delay:g}" for r in sub.itertuples())
        lines.append(f"  {w:22s} {s:11s} resid {sub.post_drift_bias.iloc[0]:+7.1%}"
                     f"  fa={int(sub.false_alarms.sum()):2d} | {vals}")
    return "\n".join(lines)
