"""Dirty-data tests. Run: python tests/test_dirty.py"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unconstrain import api
from unconstrain.prep import build_panel, read_table

RNG = np.random.default_rng(0)


def base(n_series=12, n_days=30):
    rows = []
    for s in range(n_series):
        for d in range(n_days):
            rows.append((f"SKU{s}", 100 + d, RNG.poisson(4), 0))
    return pd.DataFrame(rows, columns=["sku", "day", "qty", "oos_hours"])


def test_duplicates_are_summed():
    df = base()
    dup = pd.concat([df, df.iloc[:20]], ignore_index=True)
    panel = build_panel(dup, sales="qty", series="sku", date="day",
                        stock_hours="oos_hours")
    kinds = {i["kind"]: i["count"] for i in panel.issues}
    assert kinds.get("duplicates", 0) == 20, kinds
    assert panel.observed.all()


def test_missing_days_are_not_zero():
    df = base()
    hole = (df.sku == "SKU0") & df.day.between(105, 109)
    df = df[~hole]                                 # 5 missing days in one series
    panel = build_panel(df, sales="qty", series="sku", date="day",
                        stock_hours="oos_hours")
    kinds = {i["kind"]: i["count"] for i in panel.issues}
    assert kinds.get("unobserved", 0) == 5, kinds
    assert not panel.observed[0, 5]                # the hole is marked, not zeroed
    assert panel.observed.sum() < panel.observed.size

    dec = api.decensor(df.rename(columns={"day": "date"}), sales="qty",
                       series="sku", date="date", stock_hours="oos_hours",
                       complete=True)
    assert dec.demand_hat.isna().sum() == 5, dec.demand_hat.isna().sum()


def test_negative_sales_clipped():
    df = base()
    df.loc[:5, "qty"] = -3
    panel = build_panel(df, sales="qty", series="sku", date="day",
                        stock_hours="oos_hours")
    kinds = {i["kind"]: i["count"] for i in panel.issues}
    assert kinds.get("negative_sales", 0) == 6, kinds
    assert np.nanmin(panel.values) >= 0


def test_ragged_series():
    df = base()
    df = df[~((df.sku == "SKU1") & (df.day < 110))]
    panel = build_panel(df, sales="qty", series="sku", date="day",
                        stock_hours="oos_hours")
    kinds = {i["kind"]: i["count"] for i in panel.issues}
    assert kinds.get("ragged", 0) >= 1, kinds


def test_column_guessing():
    df = base().rename(columns={"qty": "Quantity Sold", "sku": "Item",
                                "day": "OrderDate", "oos_hours": "Hours Out"})
    rep = api.diagnose(df)
    assert rep.readiness == "R2", rep.readiness
    assert rep.n_series == 12


def test_gbk_encoding_and_bom():
    df = base()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "gbk.csv"
        df.to_csv(p, index=False, encoding="gb18030")
        back = read_table(p)
        assert len(back) == len(df)
        p2 = Path(td) / "bom.csv"
        df.to_csv(p2, index=False, encoding="utf-8-sig")
        assert len(read_table(p2)) == len(df)


def test_all_nan_series_does_not_crash():
    df = base()
    df = df[df.sku != "SKU0"]
    panel = build_panel(df, sales="qty", series="sku", date="day",
                        stock_hours="oos_hours")
    rep = api.diagnose(df)
    assert rep.n_series == 11
    dec = api.decensor(df, sales="qty", series="sku", date="day",
                       stock_hours="oos_hours")
    assert len(dec) == len(df)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("all dirty-data tests passed")
