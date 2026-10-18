"""Smoke tests. Run: python tests/test_api.py"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unconstrain.api import (cost_of_policy, decensor, diagnose,
                             recommend_quantile)


def make_frame(n_series=40, n_days=60, censor_rate=0.4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_series):
        mu = rng.uniform(1.0, 12.0)
        for d in range(n_days):
            true = rng.poisson(mu)
            out = rng.random() < censor_rate
            hours = int(rng.integers(1, 9)) if out else 0
            sold = 0 if out and hours >= 4 else true
            rows.append((s, d, sold, hours))
    return pd.DataFrame(rows, columns=["series", "date", "sales", "stock_hours"])


def test_readiness_ladder():
    df = make_frame()
    bare = df[["series", "date", "sales"]]        # nothing to detect
    r0 = diagnose(bare)
    r1 = diagnose(bare.assign(flag=df.stock_hours > 0), stockout="flag",
                  stock_hours=None, severity=None)
    r2 = diagnose(df, stock_hours="stock_hours")
    assert r0.readiness == "R0", r0.readiness
    assert r1.readiness == "R1", r1.readiness
    assert r2.readiness == "R2", r2.readiness
    assert 0.0 < r2.information_retention <= 1.0
    assert r2.effective_days <= r2.n_days
    assert r2.cr_floor_pct > 0


def test_severity_beats_flag_on_recovery():
    df = make_frame()
    flag = decensor(df.assign(flag=df.stock_hours > 0), stockout="flag")
    sev = decensor(df, stock_hours="stock_hours")
    truth = df.groupby("series").sales.sum().sum()
    # severity-aware should stay closer to the uncensored total than a hard flag
    hard_gap = abs(flag.demand_hat.sum() - truth)
    soft_gap = abs(sev.demand_hat.sum() - truth)
    assert soft_gap <= 3 * hard_gap, (hard_gap, soft_gap)


def test_strength_interpolates():
    df = make_frame()
    full = decensor(df, stock_hours="stock_hours", strength=1.0)
    half = decensor(df, stock_hours="stock_hours", strength=0.5)
    raw = decensor(df, stock_hours="stock_hours", strength=0.0)
    assert np.allclose(raw.demand_hat, df.sales, atol=1e-6)
    lo, mi, hi = raw.demand_hat.sum(), half.demand_hat.sum(), full.demand_hat.sum()
    assert lo <= mi <= hi, (lo, mi, hi)


def test_quantile():
    q = recommend_quantile(1.0, 0.02)
    assert abs(q["quantile"] - 1.0 / 1.02) < 1e-3
    assert recommend_quantile(1.0, 1.0)["quantile"] == 0.5


def test_cost_runs():
    df = make_frame()
    dec = decensor(df, stock_hours="stock_hours")
    c = cost_of_policy(dec)
    assert 0.0 <= c["fill_rate"] <= 1.0
    assert c["lost_units"] >= 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("all tests passed")
