"""Figures.  Five plots, one per claim.

English labels on purpose: the default matplotlib font stack mangles Chinese
glyphs on Windows and a boxed-out label in a saved figure is unrecoverable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .experiments import H, censor_by_policy
from .methods import HACKS, HARD_ORDER, LABELS, reconstruct
from .simulate import DemandConfig, simulate_demand

COLORS = {
    "naive": "#1b1b1b",
    "interp": "#e07a5f",
    "mean": "#f2b134",
    "seasonal": "#b08968",
    "km": "#2a9d8f",
    "tobit": "#457b9d",
    "nb": "#6a4c93",
    "oracle": "#9aa5b1",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
def plot_bias_ladder(e1, path):
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for m in ["oracle"] + list(HARD_ORDER):
        sub = e1[e1.method == m].sort_values("rate")
        if sub.empty:
            continue
        kw = dict(marker="o", ms=4, lw=2.0)
        if m == "oracle":
            kw.update(ls="--", color=COLORS["oracle"], lw=1.4)
        else:
            kw.update(color=COLORS[m], ls="--" if m in HACKS else "-")
        ax.plot(sub.rate * 100, sub.bias, label=LABELS.get(m, m), **kw)

    ax.axhline(0, color="k", lw=0.9, alpha=0.5)
    ax.set_xlabel("stock-out rate  (% of SKU-days constrained)")
    ax.set_ylabel("forecast bias vs true demand  (%)")
    ax.set_title("Every method sees the same sales -- only the reconstruction differs")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    return _save(fig, path)


# --------------------------------------------------------------------------
def plot_rank_flip(e1, path):
    """Rank by bias vs rank by the only metric you can compute.

    WAPE is an L1 loss, so it is minimised by the conditional *median*.  On
    right-skewed, zero-inflated demand the median sits well below the mean, so
    an under-forecast scores better on WAPE than the unbiased forecast does.
    The two rankings therefore disagree almost completely.
    """
    rates = np.sort(e1.rate.unique())
    rate = rates[len(rates) // 2]
    sub = e1[np.isclose(e1.rate, rate)].set_index("method")
    methods = [m for m in ["oracle"] + list(HARD_ORDER) if m in sub.index]

    rank_bias = sub.loc[methods, "bias"].abs().rank()
    rank_wape = sub.loc[methods, "wape_obs"].rank()

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for m in methods:
        ax.plot([0, 1], [rank_bias[m], rank_wape[m]], marker="o", ms=6,
                lw=2.0, color=COLORS.get(m, "#555"), alpha=0.95)
        ax.annotate(LABELS.get(m, m), (0, rank_bias[m]), xytext=(-10, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=8, color=COLORS.get(m, "#555"))
        ax.annotate(f"{rank_wape[m]:.0f}", (1, rank_wape[m]), xytext=(8, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=8, color=COLORS.get(m, "#555"))
    ax.set_xlim(-0.55, 1.25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["rank by |bias|\n(vs true demand)",
                        "rank by WAPE\n(vs observed sales)"])
    ax.set_ylim(len(methods) + 0.6, 0.4)
    ax.set_yticks(range(1, len(methods) + 1))
    ax.set_ylabel("rank  (1 = best)")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"The metric you can compute disagrees with the one that matters"
                 f"\n(at {rate:.1%} stock-outs)")
    return _save(fig, path)


# --------------------------------------------------------------------------
def plot_reconstruction(path, cfg: DemandConfig | None = None, cover: float = 4.0,
                        window: int = 70):
    cfg = cfg or DemandConfig()
    world = simulate_demand(cfg)
    dem, promo = world["demand"], world["promo"]
    sim = censor_by_policy(dem, cover)
    obs, msk = sim["sales"], sim["mask"]
    rec_km = reconstruct(obs, msk, "km", exog=promo[:, :, None])

    # a high-volume SKU that is genuinely constrained -- a slow mover capped at
    # zero says nothing interesting
    cand = [i for i in range(dem.shape[0])
            if msk[i].mean() > 0.07 and dem[i].mean() > 1.5]
    i = max(cand, key=lambda k: dem[k].mean()) if cand else 0
    first = int(np.argmax(msk[i]))
    start = max(0, min(first - window // 4, dem.shape[1] - window))
    sl = slice(start, start + window)
    t = np.arange(start, start + window)

    fig, ax = plt.subplots(figsize=(7.8, 3.9))
    ax.fill_between(t, 0, dem[i, sl], step="mid", color="#8d99ae", alpha=0.35,
                    lw=0, label="true demand (unobservable)")
    ax.step(t, dem[i, sl], where="mid", color="#8d99ae", lw=0.9)
    ax.step(t, obs[i, sl], where="mid", color="#1b1b1b", lw=1.5,
            label="observed sales")
    ax.plot(t, rec_km[i, sl], color=COLORS["km"], lw=2.0,
            label="Kaplan-Meier reconstruction")
    cap = np.where(msk[i, sl], obs[i, sl], np.nan)
    ax.scatter(t, cap, s=30, marker="v", color="#d62828", zorder=4,
               label="stock-out: sales capped at stock")
    ax.set_xlabel("day")
    ax.set_ylabel("units")
    ax.set_title(f"SKU {i}: {msk[i].mean():.0%} of days censored -- the gap is "
                 f"demand you never saw")
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper left")
    return _save(fig, path)


# --------------------------------------------------------------------------
def plot_loop(e2, path):
    """Cumulative lost units: monotone, so the divergence is not drowned in
    week-to-week noise the way a per-period stock-out rate is."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for m, sub in e2.groupby("method", sort=False):
        sub = sub.sort_values("iteration")
        color = COLORS.get(m, "#555")
        ax.plot(sub.iteration, sub.cum_lost.values, marker="o", ms=3.5, lw=2.0,
                color=color, ls="--" if "oracle" in m else "-",
                label="oracle (history uncensored)" if "oracle" in m else m)
    ax.set_xlabel("replanning period (weekly)")
    ax.set_ylabel("cumulative lost units")
    ax.set_title("A low forecast shrinks the buffer, which hides more demand")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, path)


# --------------------------------------------------------------------------
def plot_drift(e3, kept, drift_day, train_end, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    drift_i = drift_day - train_end
    censored = [k for k in kept if k.startswith("censored")]
    name = censored[min(1, len(censored) - 1)]
    streams = kept[name]["streams"]

    ax = axes[0]
    for sname, color in (("demand", "#8d99ae"), ("sales", "#d62828"),
                         ("de-censored", "#2a9d8f")):
        y = streams[sname] * 100
        base = float(np.mean(y[:max(drift_i, 1)]))
        smooth = np.convolve(y - base, np.ones(7) / 7, mode="valid")
        ax.plot(np.arange(len(y)), y - base, lw=0.4, color=color, alpha=0.2)
        ax.plot(np.arange(len(smooth)) + 6, smooth, lw=2.0, color=color,
                label=sname)
    ax.axvline(drift_i, color="k", ls=":", lw=1.2)
    ax.axhline(0, color="k", lw=0.7, alpha=0.4)
    ax.set_xlabel("days after the training window")
    ax.set_ylabel("shift in forecast error (pp, 7-day mean)")
    ax.set_title(f"One shock, three monitors -- {name.replace('censored ', '')}, "
                 f"{kept[name]['rate']:.0%} of SKU-days capped")
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax = axes[1]
    for sname, color in (("demand", "#8d99ae"), ("sales", "#d62828"),
                         ("de-censored", "#2a9d8f")):
        sub = (e3[e3.stream == sname].groupby("censoring").post_drift_bias.mean()
               .sort_index())
        ax.plot(sub.index * 100, sub.values * 100, marker="o", ms=5,
                lw=2.0, color=color, label=sname)
    ax.axhline(0, color="k", lw=0.9, alpha=0.6)
    ax.set_xlabel("stock-out rate (%)")
    ax.set_ylabel("residual after the shock (%)")
    ax.set_title("The sales signal does not just weaken -- it flips sign")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


# --------------------------------------------------------------------------
def make_all(e1, e2, e3, kept, outdir, cfg: DemandConfig | None = None,
             drift_day: int = 380, train_end: int = 300):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    made = [
        plot_bias_ladder(e1, outdir / "e1_bias_ladder.png"),
        plot_rank_flip(e1, outdir / "e2_rank_flip.png"),
        plot_reconstruction(outdir / "e3_reconstruction.png", cfg=cfg),
        plot_loop(e2, outdir / "e4_closed_loop.png"),
        plot_drift(e3, kept, drift_day, train_end, outdir / "e5_drift_blindspot.png"),
    ]
    return [str(p) for p in made]
