"""Command line entry point.

    python -m unconstrain --data real            # FreshRetailNet-50K
    python -m unconstrain --data synthetic --exp all

Real data needs ``data/train.parquet`` and ``data/eval.parquet`` (see README for
the two curl commands).  Synthetic data needs nothing.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import figures, figures_real
from .experiments import (format_e1, format_e2, format_e3, run_e1, run_e2,
                          run_e3)
from .experiments_real import (format_r1, format_r2, format_r4, format_r5,
                               run_foundation, run_quantile_study, run_real)
from .simulate import DemandConfig


def _real(args) -> None:
    from .realdata import align_holdout, describe, load_panel

    d = Path(args.data_dir)
    print(f"[real] loading {d / 'train.parquet'}")
    panel = load_panel(str(d / "train.parquet"), n_series=args.series,
                       seed=args.seed)
    print(describe(panel))
    holdout = align_holdout(panel, str(d / "eval.parquet"))
    res = run_real(panel, holdout, methods=args.methods)
    print("\n" + format_r1(res["r1"], res["r3"]))
    print("\n" + format_r2(res["r2"]))
    if not args.quick:
        res["r4"] = run_quantile_study(panel, holdout)
        print("\n" + format_r4(res["r4"]))
    if args.foundation:
        print("\n[R5] zero-shot foundation model")
        res["r5"] = run_foundation(panel, holdout)
        print("\n" + format_r5(res["r5"]))
    if not args.no_figs:
        for f in figures_real.make_all_real(res, args.out):
            print("  figure:", f)


def _synthetic(args) -> None:
    cfg = DemandConfig(seed=args.seed)
    made = []
    if args.exp in ("e1", "all"):
        print("\n[E1] bias ladder")
        e1 = run_e1(cfg)
        print("\n" + format_e1(e1))
    if args.exp in ("e2", "all"):
        print("\n[E2] closed loop")
        e2 = run_e2(cfg)
        print("\n" + format_e2(e2))
    if args.exp in ("e3", "all"):
        print("\n[E3] drift")
        e3, kept = run_e3(cfg)
        print("\n" + format_e3(e3))
    if not args.no_figs and args.exp == "all":
        made = figures.make_all(e1, e2, e3, kept, args.out, cfg=cfg)
        for m in made:
            print("  figure:", m)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="unconstrain",
        description="Demand you never fully observed: bias, cost, monitoring.")
    p.add_argument("--data", default="real", choices=["real", "synthetic"],
                   help="real FreshRetailNet panel, or the controlled simulator")
    p.add_argument("--exp", default="all", choices=["e1", "e2", "e3", "all"],
                   help="which synthetic experiment to run")
    p.add_argument("--series", type=int, default=600,
                   help="how many store-product series to load (real data)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-figs", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="skip the quantile study (real data)")
    p.add_argument("--foundation", action="store_true",
                   help="also run the zero-shot Chronos comparison (needs "
                        "chronos-forecasting + transformers, HF_ENDPOINT)")
    args = p.parse_args(argv)
    args.methods = ("naive", "interp", "impute", "km", "tobit", "nb",
                    "km_s", "tobit_s", "nb_s")

    t0 = time.time()
    _real(args) if args.data == "real" else _synthetic(args)
    print(f"\ndone in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
