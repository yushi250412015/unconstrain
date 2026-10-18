"""One self-contained HTML report.

The terminal output is for the person doing the analysis. This is for the person
who signs the purchase order: a single file with no dependencies, no server, no
plot library — open it, read it, forward it.
"""

from __future__ import annotations

import html
from datetime import datetime

import numpy as np
import pandas as pd

from .api import decensor, diagnose
from .prep import build_panel

_CSS = """
body{font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  color:#1b1b1b;max-width:880px;margin:40px auto;padding:0 22px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:15px;margin:34px 0 10px;
  text-transform:uppercase;letter-spacing:.06em;color:#666}
.sub{color:#666;margin:0 0 26px}
table{border-collapse:collapse;width:100%;margin:8px 0 4px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid #e8e8e8}
th{font-weight:600;font-size:12px;color:#666;text-transform:uppercase;
  letter-spacing:.04em}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.badge{display:inline-block;padding:2px 9px;border-radius:11px;font-weight:600;
  font-size:13px}
.r0{background:#f4d6d6;color:#8a1c1c} .r1{background:#fbe5c8;color:#8a551c}
.r2{background:#d3ecd9;color:#1c6b39} .r3{background:#cfe3f4;color:#1c4f7a}
.big{font-size:26px;font-weight:600;font-variant-numeric:tabular-nums}
.card{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:14px 0}
.card div{border:1px solid #e8e8e8;border-radius:8px;padding:12px 14px}
.card span{display:block;font-size:12px;color:#666;margin-bottom:3px}
.act{border-left:3px solid #1c6b39;padding:6px 0 6px 14px;margin:10px 0}
.warn{border-left:3px solid #b5651c;padding:6px 0 6px 14px;margin:10px 0}
pre{background:#f6f6f6;padding:10px 12px;border-radius:6px;overflow:auto;
  font-size:12px}
footer{margin-top:40px;color:#999;font-size:12px}
"""

_RUNGS = [
    ("R0", "sales only", "censoring unidentifiable"),
    ("R1", "+ stock-out flag", "correction over-shoots 9-28%"),
    ("R2", "+ stock hours", "correction lands near 0"),
    ("R3", "+ hourly sales", "same, self-calibrated"),
]


def _fmt_pct(x, digits=1):
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def build_report(df: pd.DataFrame, *, sales="sales", series="series", date="date",
                 stockout=None, severity=None, stock_hours=None,
                 open_hours: int = 16, title: str = "Censoring audit") -> str:
    rep = diagnose(df, sales=sales, series=series, date=date, stockout=stockout,
                   severity=severity, stock_hours=stock_hours,
                   open_hours=open_hours)

    # three recovery scenarios, on the same rows
    scenarios = []
    try:
        m = decensor(df, sales=sales, series=series, date=date,
                     severity=severity, stock_hours=stock_hours,
                     stockout=stockout, open_hours=open_hours, method="tobit")
        lift = 100 * (m["demand_hat"].sum() / df[sales].sum() - 1)
        scenarios.append(("severity-weighted (recommended)", lift))
    except Exception:  # noqa: BLE001
        pass
    if stock_hours or severity:
        try:
            cols = [c for c in (series, date, sales) if c is not None]
            flag = df[cols].copy()
            panel = build_panel(df, sales=sales, series=series, date=date,
                                stock_hours=stock_hours, open_hours=open_hours)
            flat = pd.DataFrame({
                series: np.repeat(np.asarray(panel.series), panel.n_days),
                date: np.tile(pd.to_datetime(panel.dates), panel.n_series),
                "out_of_stock": (panel.mask & panel.observed).reshape(-1),
            })
            src = df.copy()
            src[date] = pd.to_datetime(src[date], errors="coerce")
            fd = src.merge(flat, on=[series, date], how="left")
            for strength, label in ((1.0, "binary flag, full strength"),
                                    (0.4, "binary flag, strength 0.4")):
                got = decensor(fd, sales=sales, series=series, date=date,
                               stockout="out_of_stock", method="tobit",
                               strength=strength)
                scenarios.append((label, 100 * (got["demand_hat"].sum()
                                                / df[sales].sum() - 1)))
        except Exception:  # noqa: BLE001
            pass

    rows = "\n".join(
        f"<tr><td>{r}</td><td>{html.escape(desc)}</td><td>{html.escape(note)}</td>"
        f"<td>{'&#9679; here' if r == rep.readiness else ''}</td></tr>"
        for r, desc, note in _RUNGS)

    issues = "".join(f"<li>{html.escape(i)}</li>" for i in rep.data_issues) \
        or "<li>none found</li>"
    actions = "".join(f"<div class='act'>{i + 1}. {html.escape(a)}</div>"
                      for i, a in enumerate(rep.actions))

    scen_html = "".join(
        f"<tr><td>{html.escape(n)}</td><td class='num'>{v:+.1f}%</td></tr>"
        for n, v in scenarios) or "<tr><td colspan=2>no correction available</td></tr>"

    risk = f"{100 * rep.volume_at_risk:.1f}%" if rep.volume_at_risk else "n/a"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="sub">{rep.n_series} series &middot; {rep.n_days} days &middot;
{rep.coverage:.1%} observed &middot; generated {datetime.now():%Y-%m-%d}</p>

<span class="badge {rep.readiness.lower()}">readiness {rep.readiness}</span>
&nbsp; the correction you can justify today

<h2>What the censoring costs</h2>
<div class="card">
  <div><span>censored days</span><div class="big">{100 * rep.censoring_rate:.0f}%</div></div>
  <div><span>volume at risk</span><div class="big">{risk}</div></div>
  <div><span>level error floor</span><div class="big">{rep.cr_floor_pct:.1f}%</div></div>
</div>
<p>Only {rep.information_retention:.0%} of the demand-level information survives
the censoring: {rep.effective_days:.0f} effective days out of {rep.n_days}.
No method can estimate the level better than {rep.cr_floor_pct:.1f}% from this
history, so the remaining error is not an engineering problem.</p>

<h2>What each correction would recover</h2>
<table><tr><th>method</th><th style="text-align:right">volume recovered</th></tr>
{scen_html}</table>
<p>The full-strength flag overshoots the volume at risk; the severity-weighted
estimate is deliberately conservative. Both are honest; the difference is what
your data can support.</p>

<h2>Where you are on the ladder</h2>
<table><tr><th>rung</th><th>you log</th><th>consequence</th><th></th></tr>{rows}</table>

<h2>Data issues found</h2><ul>{issues}</ul>

<h2>Next steps</h2>{actions}

<footer>generated by <code>unconstrain</code> &middot; bias estimates measured on
FreshRetailNet-50K, scored on fully-stocked days only</footer>
</body></html>"""
