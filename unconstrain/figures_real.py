"""Figures for the real-data half.  English labels, same reason as before."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})

GROUPS = {
    "naive": "#1b1b1b", "interp": "#e07a5f",
    "km": "#8ecae6", "tobit": "#457b9d", "nb": "#2a6f97",
    "km_s": "#95d5b2", "tobit_s": "#2a9d8f", "nb_s": "#1b6b52",
}


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_r1(r1, path):
    df = r1.set_index("method")
    order = [m for m in GROUPS if m in df.index]
    x = np.arange(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    ax = axes[0]
    vals = df.loc[order, "bias"]
    ax.bar(x, vals, color=[GROUPS[m] for m in order], edgecolor="white")
    ax.axhline(0, color="k", lw=1.0)
    for i, v in zip(x, vals):
        ax.annotate(f"{v:+.1f}", (i, v), xytext=(0, 4 if v >= 0 else -12),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylabel("bias vs true demand (%)")
    ax.set_title("Bias on the 7-day holdout\n"
                 "(pale = textbook assumption, dark = severity-weighted)")

    ax = axes[1]
    vals = df.loc[order, "wape"]
    ax.bar(x, vals, color=[GROUPS[m] for m in order], edgecolor="white")
    for i, v in zip(x, vals):
        ax.annotate(f"{v:.1f}", (i, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylabel("WAPE (%)")
    ax.set_title("Accuracy moves the same way")
    return _save(fig, path)


def plot_r2(r2, path):
    """Binned bias vs stock-out rate: the practical lookup table."""
    curves = r2.attrs["curves"]
    picks = [m for m in ("naive", "tobit", "tobit_s") if m in curves]
    styles = {"naive": dict(color="#1b1b1b", ls="--", marker="o"),
              "tobit": dict(color="#8ecae6", ls="-", marker="s"),
              "tobit_s": dict(color="#2a9d8f", ls="-", marker="o"),
              "nb": dict(color="#457b9d", ls="-", marker="s"),
              "nb_s": dict(color="#1b6b52", ls="-", marker="o")}

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for m in picks:
        xs, ys = curves[m]
        row = r2[r2.method == m].iloc[0]
        ax.plot(xs * 100, ys * 100, lw=2.0, ms=6, label=m, **styles.get(m, {}))
    ax.axhline(0, color="k", lw=0.9, alpha=0.6)
    ax.set_xlabel("stock-out rate during training (%)")
    ax.set_ylabel("forecast bias on clean days (%)")
    ax.set_title("Availability in, bias out:\n"
                 "the textbook correction sits 6-14 points high at every "
                 "availability level")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def plot_r3(r3, path):
    df = r3.sort_values("lost")
    order = list(df.method)
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    y = np.arange(len(order))
    ax.barh(y, df.lost.values, color=[GROUPS.get(m, "#555") for m in order])
    for i, (lost, fill) in enumerate(zip(df.lost.values, df.fill_rate.values)):
        ax.annotate(f"{lost:.0f} units   fill {fill:.1%}", (lost, i),
                    xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("units of demand lost under the same policy (7 days)")
    ax.set_title("Same policy, same week -- the ordering is decided by the forecast")
    ax.set_xlim(0, df.lost.max() * 1.45)
    return _save(fig, path)


def plot_r4(r4, path):
    """Lost units as a function of the ordering quantile."""
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    for m, sub in r4.groupby("method", sort=False):
        sub = sub.sort_values("q")
        ax.plot(sub.q, sub.lost, marker="o", ms=5, lw=2.0,
                color=GROUPS.get(m, "#555"), label=m)
    ax.set_xlabel("ordering quantile  (newsvendor optimum for 50:1 is ~0.98)")
    ax.set_ylabel("units of demand lost (7 days)")
    ax.set_title("Ordering the mean is the wrong target even when it is unbiased")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def plot_r5(r5, r1, path):
    """Foundation model, with and without a de-censored history."""
    gbm = r1.set_index("method")
    bars = []
    for r in r5.itertuples():
        bars.append((f"Chronos\n[{r.history}]", r.wape, "#d62828"))
    for m, col in (("naive", "#1b1b1b"), ("tobit_s", "#2a9d8f")):
        if m in gbm.index:
            bars.append((f"GBM\n[{m}]", gbm.loc[m, "wape"], col))

    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    x = np.arange(len(bars))
    ax.bar(x, [b[1] for b in bars], color=[b[2] for b in bars], edgecolor="white")
    for i, b in enumerate(bars):
        ax.annotate(f"{b[1]:.1f}", (i, b[1]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylabel("WAPE on clean holdout days (%)")
    ax.set_ylim(0, max(b[1] for b in bars) * 1.18)
    ax.set_title("A foundation model is only as good as the history it is fed")
    return _save(fig, path)


def make_all_real(res: dict, outdir) -> list[str]:
    outdir = Path(outdir)
    made = [
        plot_r1(res["r1"], outdir / "r1_soft_vs_hard.png"),
        plot_r2(res["r2"], outdir / "r2_bias_law.png"),
        plot_r3(res["r3"], outdir / "r3_hidden_cost.png"),
    ]
    if res.get("r4") is not None:
        made.append(plot_r4(res["r4"], outdir / "r4_quantile.png"))
    if res.get("r5") is not None:
        made.append(plot_r5(res["r5"], res["r1"], outdir / "r5_foundation.png"))
    return made
