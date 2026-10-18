"""Read and repair a real sales export.

Every function here exists because a real file broke something:

    encoding        a GBK export read as UTF-8 turns headers into mojibake
    column names    "Qty", "qty", "Quantity Sold", "销量" are the same column
    duplicates      transaction-level exports have many rows per SKU-day
    missing dates   a SKU with no row on a day is not a zero-sale day
    negatives       returns and corrections arrive as negative units
    ragged series   products launch mid-window and discontinue before the end

The last two are the dangerous ones. Treating an unobserved day as a zero
teaches the model that demand collapsed; the panel therefore carries an explicit
``observed`` flag and the API never reconstructs a cell that was not there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SALES_HINTS = ("sales", "sale", "qty", "quantity", "units", "demand", "amount",
               "volume", "sold", "销量", "销售", "数量")
SERIES_HINTS = ("series", "sku", "item", "product", "article", "id", "商品", "编码")
DATE_HINTS = ("date", "dt", "day", "time", "日期", "时间")
FLAG_HINTS = ("stockout", "out_of_stock", "oos", "stock_out", "缺货标记", "缺货")
HOURS_HINTS = ("hours_out", "stock_hours", "oos_hours", "hours", "缺货小时", "缺货时长")


@dataclass
class Panel:
    values: np.ndarray            # (n, T) sales, NaN where not observed
    observed: np.ndarray          # (n, T) bool
    mask: np.ndarray              # (n, T) bool, censored
    severity: np.ndarray | None    # (n, T) in [0, 1] or None
    series: list
    dates: list
    issues: list[dict] = field(default_factory=list)

    @property
    def n_series(self) -> int:
        return self.values.shape[0]

    @property
    def n_days(self) -> int:
        return self.values.shape[1]

    def summary(self) -> str:
        return "\n".join(f"  {i['kind']:14s} {i['count']:>7d}  {i['detail']}"
                         for i in self.issues) or "  (clean)"


def read_table(path: str | Path) -> pd.DataFrame:
    """CSV/parquet with encoding fallback; the first row is the header."""
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    last = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            return pd.read_csv(p, encoding=enc)
        except UnicodeDecodeError as exc:
            last = exc
    raise last  # pragma: no cover


def _find(columns, hints, override=None):
    # an override only counts if the column actually exists: the API defaults
    # are "sales"/"series"/"date", which are absent from most real exports
    if override and override in columns:
        return override
    low = {str(c).strip().lower(): c for c in columns}
    for c in low:
        if any(h in c for h in hints):
            return low[c]
    return None


def guess_columns(df: pd.DataFrame, **overrides) -> dict:
    """Best-effort mapping from a raw export to the four columns we need."""
    out = {
        "sales": _find(df.columns, SALES_HINTS, overrides.get("sales")),
        "series": _find(df.columns, SERIES_HINTS, overrides.get("series")),
        "date": _find(df.columns, DATE_HINTS, overrides.get("date")),
        "stockout": _find(df.columns, FLAG_HINTS, overrides.get("stockout")),
        "stock_hours": _find(df.columns, HOURS_HINTS, overrides.get("stock_hours")),
        "severity": _find(df.columns, ("severity", "censor", "严重度"),
                          overrides.get("severity")),
    }
    return out


def build_panel(df: pd.DataFrame, *, sales: str, series: str, date: str,
                stockout: str | None = None, stock_hours: str | None = None,
                severity: str | None = None, open_hours: int = 16) -> Panel:
    """Long table -> rectangular panel, with every repair logged."""
    issues: list[dict] = []
    keep = [(series, "series"), (date, "date"), (sales, "sales")]
    if stockout:
        keep.append((stockout, "stockout"))
    if stock_hours:
        keep.append((stock_hours, "stock_hours"))
    if severity:
        keep.append((severity, "severity"))
    d = df[[c for c, _ in keep]].copy()
    d.columns = [n for _, n in keep]

    # --- types -----------------------------------------------------------
    before = len(d)
    d["sales"] = pd.to_numeric(d["sales"], errors="coerce")
    bad = int(d["sales"].isna().sum())
    if bad:
        issues.append(dict(kind="non_numeric", count=bad,
                           detail="rows whose sales could not be parsed; dropped"))
        d = d[d["sales"].notna()]
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    bad_d = int(d["date"].isna().sum())
    if bad_d:
        issues.append(dict(kind="bad_date", count=bad_d, detail="dated rows dropped"))
        d = d[d["date"].notna()]
    if len(d) < before:
        pass

    neg = int((d["sales"] < 0).sum())
    if neg:
        issues.append(dict(kind="negative_sales", count=neg,
                           detail="returns/corrections clipped to 0"))
        d["sales"] = d["sales"].clip(lower=0)

    # --- duplicates ------------------------------------------------------
    dup = int(d.duplicated(subset=["series", "date"]).sum())
    if dup:
        issues.append(dict(kind="duplicates", count=dup,
                           detail="same series-day appears more than once; summed"))
    agg = {"sales": "sum"}
    for c in ("stockout", "stock_hours", "severity"):
        if c in d.columns:
            agg[c] = "max"
    d = d.groupby(["series", "date"], as_index=False).agg(agg)

    # --- rectangle -------------------------------------------------------
    piv = d.pivot(index="series", columns="date", values="sales")
    dates = list(piv.columns)
    sids = list(piv.index)
    values = piv.to_numpy(dtype=float)
    observed = ~np.isnan(values)

    total_cells = values.size
    missing = int(total_cells - observed.sum())
    if missing:
        issues.append(dict(kind="unobserved", count=missing,
                           detail=f"{missing / total_cells:.1%} of series-days have "
                                  "no row; they are NOT treated as zero sales"))

    # ragged series (start late / end early)
    first = np.argmax(observed, axis=1)
    last = observed.shape[1] - 1 - np.argmax(observed[:, ::-1], axis=1)
    ragged = int(np.sum((first > 0) | (last < observed.shape[1] - 1)))
    if ragged:
        issues.append(dict(kind="ragged", count=ragged,
                           detail="series that start late or end early"))

    # --- censoring -------------------------------------------------------
    mask = np.zeros_like(observed)
    sev = None
    if stockout and "stockout" in d.columns:
        m = d.pivot(index="series", columns="date", values="stockout")
        mask = m.reindex(index=sids, columns=dates).to_numpy() > 0.5
        mask = np.where(observed, mask, False)
    if stock_hours and "stock_hours" in d.columns:
        h = d.pivot(index="series", columns="date", values="stock_hours")
        hh = np.nan_to_num(h.reindex(index=sids, columns=dates).to_numpy())
        mask = (hh > 0) & observed
        sev = np.clip(hh / float(open_hours), 0.0, 1.0)
    if severity and "severity" in d.columns:
        s = d.pivot(index="series", columns="date", values="severity")
        sev = np.clip(np.nan_to_num(s.reindex(index=sids, columns=dates).to_numpy()), 0, 1)
        mask = mask | ((sev > 0) & observed)
    if sev is not None:
        sev = np.where(mask, sev, 0.0)

    odd = int(np.sum((sev > 1.0))) if sev is not None else 0
    if odd:
        issues.append(dict(kind="severity_gt1", count=odd, detail="clipped to 1"))

    return Panel(values=values, observed=observed, mask=mask, severity=sev,
                 series=sids, dates=dates, issues=issues)


def fill_gaps(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Interpolate unobserved cells *for the reconstruction only*.

    The model needs a value to compute lags; the output keeps those cells as
    NaN so nothing is fabricated downstream.
    """
    out = values.copy()
    idx = np.arange(values.shape[1])
    for i in range(values.shape[0]):
        good = observed[i] & np.isfinite(values[i])
        if good.sum() == 0:
            out[i] = 0.0
        elif good.sum() == 1:
            out[i] = values[i][good][0]
        else:
            out[i] = np.interp(idx, idx[good], values[i][good])
    return np.nan_to_num(out, nan=0.0)
