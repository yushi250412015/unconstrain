"""Real data: FreshRetailNet-50K (Dingdong / arXiv:2505.16319).

Why this dataset: it is the only public retail benchmark that ships an
*annotation for the censoring itself* -- hourly stock status, not just sales.
Without that annotation there is no way to know whether a zero is "no customer"
or "no stock", and every method here would be unfalsifiable.

Protocol note (important): the dataset does not contain latent demand, because
demand during a stock-out is by definition unobserved.  So the real-data
evaluation only ever scores days that were **fully stocked**, where sales and
demand coincide.  That keeps the test honest: no invented ground truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: the dataset annotates stock status over hours 6..21 inclusive (16 hours)
BUSINESS_HOURS = 16
HOUR_LO, HOUR_HI = 6, 22

COLS = [
    "store_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt",
    "discount", "holiday_flag", "activity_flag", "precpt",
    "avg_temperature", "avg_humidity", "avg_wind_level",
]


def _read(path, columns=COLS) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=columns)
    df["sid"] = df["store_id"].astype(np.int64) * 1_000_000 + df["product_id"]
    df["day"] = df.groupby("sid")["dt"].rank(method="dense").astype(np.int32) - 1
    return df


def select_series(df: pd.DataFrame, n_series: int, top_pool: int = 4000,
                  seed: int = 0) -> np.ndarray:
    """Pick traded SKUs from the heavy end, then a random subset of them.

    Taking the very top by volume would concentrate every series in a handful
    of store-product pairs; sampling inside a heavy-volume pool keeps the panel
    diverse while staying away from the near-empty tail, where a 0/1 sales
    series says nothing about censoring.
    """
    vol = df.groupby("sid")["sale_amount"].sum()
    pool = vol.sort_values(ascending=False).head(top_pool).index.to_numpy()
    rng = np.random.default_rng(seed)
    n = min(int(n_series), pool.size)
    return np.sort(rng.choice(pool, size=n, replace=False))


def _to_matrix(df, value, sids, n_days) -> np.ndarray:
    m = df.pivot_table(index="sid", columns="day", values=value, aggfunc="first")
    m = m.reindex(index=sids, columns=range(n_days))
    return m.to_numpy(dtype=float)


def _load_hours(path: str, sids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hourly sales and stock status as (series, day, hour) arrays."""
    df = pd.read_parquet(path, columns=["store_id", "product_id", "dt",
                                        "hours_sale", "hours_stock_status"])
    df["sid"] = df["store_id"].astype(np.int64) * 1_000_000 + df["product_id"]
    df["day"] = df.groupby("sid")["dt"].rank(method="dense").astype(np.int32) - 1
    df = df[df["sid"].isin(sids)].sort_values(["sid", "day"])
    n = len(sids)
    t = int(df["day"].max()) + 1
    hours = np.stack(df["hours_sale"].values).reshape(n, t, 24).astype(float)
    status = np.stack(df["hours_stock_status"].values).reshape(n, t, 24).astype(float)
    return hours, status


def volume_severity(hours: np.ndarray, status: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """Share of a day's *volume* that was unreachable, not share of hours.

    Sales are not spread evenly across the day: hour 9 carries ~40x the volume
    of hour 20 in this dataset.  A one-hour stock-out at 9am therefore destroys
    far more demand than one at 8pm, and counting hours treats them as equal.
    The hourly profile is estimated per series from its fully-stocked days, so
    the weights are the SKU's own shopping pattern.
    """
    n = hours.shape[0]
    prof = np.zeros((n, 24))
    glob = hours[clean].sum(axis=0)
    glob = glob / max(glob.sum(), 1e-9)
    for i in range(n):
        sel = clean[i]
        if sel.sum() >= 5 and hours[i][sel].sum() > 0:
            row = hours[i][sel].sum(axis=0)
            prof[i] = row / row.sum()
        else:
            prof[i] = glob
    blocked = status[:, :, HOUR_LO:HOUR_HI]
    s = np.einsum("ith,ih->it", blocked, prof[:, HOUR_LO:HOUR_HI])
    return np.clip(s, 0.0, 1.0)


def load_panel(path: str, n_series: int = 600, seed: int = 0,
               top_pool: int = 4000) -> dict:
    """Load a store-product panel as dense (series x day) matrices."""
    df = _read(path)
    sids = select_series(df, n_series, top_pool, seed)
    df = df[df["sid"].isin(sids)]
    n_days = int(df["day"].max()) + 1

    sales = np.nan_to_num(_to_matrix(df, "sale_amount", sids, n_days))
    stock_h = np.nan_to_num(_to_matrix(df, "stock_hour6_22_cnt", sids, n_days))
    discount = _to_matrix(df, "discount", sids, n_days)

    mask = stock_h > 0
    hours, status = _load_hours(path, sids)
    sev_vol = volume_severity(hours, status, ~mask)

    return {
        "sales": sales,                 # observed sales == demand on clean days
        "mask": mask,                   # any stock-out hour that day
        "severity": np.where(mask, sev_vol, 0.0),      # volume weighted
        "severity_hours": np.where(mask, stock_h / BUSINESS_HOURS, 0.0),
        "promo": (np.nan_to_num(discount, nan=1.0) < 0.999).astype(float),
        "discount": np.nan_to_num(discount, nan=1.0),
        "holiday": np.nan_to_num(_to_matrix(df, "holiday_flag", sids, n_days)),
        "activity": np.nan_to_num(_to_matrix(df, "activity_flag", sids, n_days)),
        "precpt": np.nan_to_num(_to_matrix(df, "precpt", sids, n_days)),
        "temp": np.nan_to_num(_to_matrix(df, "avg_temperature", sids, n_days)),
        "humidity": np.nan_to_num(_to_matrix(df, "avg_humidity", sids, n_days)),
        "sids": sids,
        "dates": sorted(df["dt"].unique()),
        "hour_profile": hours,
    }


def align_holdout(train: dict, eval_path: str) -> dict:
    """Load the official 7-day holdout and align it to the training series."""
    df = _read(eval_path)
    df["day"] = df.groupby("sid")["dt"].rank(method="dense").astype(np.int32) - 1
    sids = train["sids"]
    n_days = int(df["day"].max()) + 1
    df = df[df["sid"].isin(sids)]
    sales = np.nan_to_num(_to_matrix(df, "sale_amount", sids, n_days))
    stock_h = np.nan_to_num(_to_matrix(df, "stock_hour6_22_cnt", sids, n_days))
    disc = np.nan_to_num(_to_matrix(df, "discount", sids, n_days), nan=1.0)
    return {
        "sales": sales,
        "mask": stock_h > 0,
        "severity": np.clip(stock_h / BUSINESS_HOURS, 0.0, 1.0),
        "promo": (disc < 0.999).astype(float),
        "discount": disc,
    }


def describe(panel: dict) -> str:
    sales, mask = panel["sales"], panel["mask"]
    clean = ~mask
    per_series = mask.mean(axis=1)
    lines = [
        f"series {sales.shape[0]}   days {sales.shape[1]}",
        f"stock-out days {mask.mean():.1%}   fully-stocked days {clean.mean():.1%}",
        f"mean daily sales {sales.mean():.3f}   on clean days {sales[clean].mean():.3f}",
        f"per-series stock-out rate: median {np.median(per_series):.2f}  "
        f"p10 {np.quantile(per_series, .1):.2f}  p90 {np.quantile(per_series, .9):.2f}",
        f"zero-sales days {(sales == 0).mean():.1%}",
    ]
    return "\n".join(lines)
