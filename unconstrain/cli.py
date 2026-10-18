"""Command line interface.

    unconstrain diagnose  sales.csv --stock-hours hours_out
    unconstrain decensor  sales.csv -o demand.csv --strength 0.4
    unconstrain quantile  --shortage 1 --holding 0.02
    unconstrain demo

The demo needs ``data/train.parquet`` (see README) and runs the whole ladder on
real data without writing anything to disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .api import cost_of_policy, decensor, diagnose, recommend_quantile


def _add_cols(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sales", default="sales", help="sales column")
    p.add_argument("--series", default="series", help="series id column")
    p.add_argument("--date", default="date", help="date column")
    p.add_argument("--stockout", default=None, help="boolean stock-out column")
    p.add_argument("--severity", default=None,
                   help="column in [0,1]: share of the day's volume unreachable")
    p.add_argument("--stock-hours", default=None,
                   help="hours out of stock that day (implies severity)")
    p.add_argument("--open-hours", type=int, default=16,
                   help="selling hours per day, for --stock-hours (default 16)")


def _read(path: str) -> pd.DataFrame:
    from .prep import read_table
    return read_table(path)


def cmd_diagnose(args) -> int:
    df = _read(args.file)
    rep = diagnose(df, sales=args.sales, series=args.series, date=args.date,
                   stockout=args.stockout, severity=args.severity,
                   stock_hours=args.stock_hours, open_hours=args.open_hours)
    if args.json:
        print(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(rep)
    return 0


def cmd_decensor(args) -> int:
    df = _read(args.file)
    out = decensor(df, sales=args.sales, series=args.series, date=args.date,
                   stockout=args.stockout, severity=args.severity,
                   stock_hours=args.stock_hours, open_hours=args.open_hours,
                   method=args.method, strength=args.strength)
    before = df[args.sales].sum()
    after = out["demand_hat"].sum()
    out.to_csv(args.out, index=False)
    lift = 100 * (after / max(before, 1e-9) - 1)
    print(f"wrote {args.out}: {len(out):,} rows, "
          f"recovered volume {lift:+.1f}% (method={args.method}, "
          f"strength={args.strength})")
    return 0


def cmd_quantile(args) -> int:
    r = recommend_quantile(args.shortage, args.holding,
                           service_floor=args.service_floor)
    print(json.dumps(r, indent=2))
    return 0


def cmd_report(args) -> int:
    from .report import build_report
    df = _read(args.file)
    html = build_report(df, sales=args.sales, series=args.series, date=args.date,
                        stockout=args.stockout, severity=args.severity,
                        stock_hours=args.stock_hours, open_hours=args.open_hours,
                        title=args.title)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"wrote {args.out} ({len(html):,} bytes) - open it in a browser")
    return 0


def cmd_demo(args) -> int:
    d = Path(args.data_dir)
    src = d / "train.parquet"
    if not src.exists():
        print(f"missing {src}; download it first (see README, two curl commands)")
        return 2

    from .realdata import align_holdout, load_panel
    panel = load_panel(str(src), n_series=args.series)
    sales, mask, sev = panel["sales"], panel["mask"], panel["severity"]
    n, T = sales.shape
    sids = np.repeat(np.arange(n), T)
    days = np.tile(np.arange(T), n)
    flat = pd.DataFrame({
        "series": sids, "date": days,
        "sales": sales.reshape(-1),
        "stock_hours": (mask * (sev * 16)).reshape(-1).round(),
    })

    print("=" * 68)
    print("RUNG R2  (sales + stock hours)")
    print("=" * 68)
    print(diagnose(flat, stock_hours="stock_hours"))

    print()
    print("=" * 68)
    print("RUNG R1  (sales + a binary flag)  -- what most teams actually have")
    print("=" * 68)
    flag_only = flat.drop(columns=["stock_hours"]).assign(
        out_of_stock=(flat["stock_hours"] > 0).to_numpy())
    print(diagnose(flag_only, stockout="out_of_stock"))

    print()
    print("=" * 68)
    print("CORRECTION")
    print("=" * 68)
    dec = decensor(flat, stock_hours="stock_hours", method="tobit")
    lift = 100 * (dec["demand_hat"].sum() / flat["sales"].sum() - 1)
    print(f"severity-weighted tobit recovers {lift:+.1f}% of volume")
    half = decensor(flag_only, stockout="out_of_stock", method="tobit", strength=0.4)
    lift_h = 100 * (half["demand_hat"].sum() / flat["sales"].sum() - 1)
    full = decensor(flag_only, stockout="out_of_stock", method="tobit", strength=1.0)
    lift_f = 100 * (full["demand_hat"].sum() / flat["sales"].sum() - 1)
    print(f"binary flag at full strength recovers {lift_f:+.1f}% (over-corrects); "
          f"strength 0.4 gives {lift_h:+.1f}%")

    print()
    print("=" * 68)
    print("DECISION")
    print("=" * 68)
    q = recommend_quantile(1.0, 0.02)
    print(json.dumps(q, indent=2))
    print("  policy cost on corrected demand:", cost_of_policy(dec))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="unconstrain",
                                description="Audit and undo demand censoring.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diagnose", help="audit a sales table")
    d.add_argument("file")
    _add_cols(d)
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_diagnose)

    c = sub.add_parser("decensor", help="write a corrected demand table")
    c.add_argument("file")
    _add_cols(c)
    c.add_argument("-o", "--out", required=True)
    c.add_argument("--method", default="tobit",
                   choices=["tobit", "nb", "km", "interp", "mean"])
    c.add_argument("--strength", type=float, default=1.0,
                   help="blend 0 (raw sales) .. 1 (full correction); use ~0.4 "
                        "when you only have a binary flag")
    c.set_defaults(func=cmd_decensor)

    q = sub.add_parser("quantile", help="newsvendor critical fractile")
    q.add_argument("--shortage", type=float, required=True)
    q.add_argument("--holding", type=float, required=True)
    q.add_argument("--service-floor", type=float, default=None)
    q.set_defaults(func=cmd_quantile)

    r = sub.add_parser("report", help="write a self-contained HTML report")
    r.add_argument("file")
    _add_cols(r)
    r.add_argument("-o", "--out", default="censor-report.html")
    r.add_argument("--title", default="Censoring audit")
    r.set_defaults(func=cmd_report)

    m = sub.add_parser("demo", help="run the whole ladder on the benchmark data")
    m.add_argument("--data-dir", default="data")
    m.add_argument("--series", type=int, default=300)
    m.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
