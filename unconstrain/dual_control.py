"""The mask is endogenous: your stocking policy decides how much you learn.

Every de-censoring paper treats the stock-out mask as a given input to be
repaired. But the mask is produced by the policy itself: stock out, and that
day's demand observation is deleted. A lean policy therefore buys lower holding
cost with *less information*, and the price of that information is never quoted.

This module quotes it. A Bayesian agent knows the demand family but not the level
of each SKU, and learns from censored sales. Because ordering *more* trivially
improves service under an asymmetric cost, the informative comparison is not
"order more vs order less" but **"same expected inventory, different variance"**:

    greedy   order the newsvendor quantile of the current posterior
    dither   same mean, but the order-up-to level is randomised (+-40%)
    explore  same, plus a positive exploration bonus

If ``dither`` learns the level faster than ``greedy`` at the same average
inventory, then information has a price of zero and the mask is a *choice*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import nbinom

SIZE = 60.0
DAYS = 7


@dataclass
class Config:
    n_sku: int = 300
    weeks: int = 30
    seeds: tuple[int, ...] = (0, 1, 2)
    holding: float = 0.02
    shortage: float = 1.0
    explore_z: float = 1.0
    dither_sd: float = 0.4


def _posterior_update(w: np.ndarray, y: np.ndarray, stock: np.ndarray,
                      grid: np.ndarray) -> np.ndarray:
    """Bayes update from one week of censored sales, per SKU.

    A day with ``sales == stock_i`` is censored: its likelihood is
    ``P(D >= stock_i)``, not ``p(sales)``.
    """
    mu = grid[None, :]
    p = SIZE / (SIZE + mu)
    s = np.asarray(stock, float).reshape(-1, 1)
    ll = np.zeros_like(w)
    for d in range(y.shape[1]):
        yd = y[:, d][:, None]
        cens = yd >= s - 1e-9
        ll_obs = nbinom.logpmf(np.minimum(yd, s), SIZE, p)
        ll_cen = nbinom.logsf(s - 1, SIZE, p)
        ll += np.where(cens, ll_cen, ll_obs)
    ll = np.nan_to_num(ll, nan=-50.0, neginf=-50.0, posinf=0.0)
    post = np.log(np.maximum(w, 1e-300)) + ll
    post -= post.max(axis=1, keepdims=True)
    out = np.exp(post)
    return out / out.sum(axis=1, keepdims=True)


def _quantile(grid: np.ndarray, w: np.ndarray, q: float) -> np.ndarray:
    cdf = np.cumsum(w, axis=1)
    idx = np.argmax(cdf >= q, axis=1)
    return grid[idx]


def run(cfg: Config = Config(), q: float = 0.91, policy: str = "greedy") -> dict:
    res = {"policy": policy, "q": q, "err": [], "cost": [], "fill": [], "inv": []}
    for seed in cfg.seeds:
        rng = np.random.default_rng(seed)
        mu_true = rng.lognormal(mean=1.0, sigma=0.7, size=cfg.n_sku)
        grid = np.geomspace(0.25, 60.0, 160)
        prior = np.exp(-0.5 * ((np.log(grid) - 1.0) / 0.9) ** 2)
        w = np.tile(prior / prior.sum(), (cfg.n_sku, 1))

        cum_cost = units = cum_lost = 0.0
        errs = []
        for _ in range(cfg.weeks):
            level = _quantile(grid, w, q)
            if policy == "greedy":
                stock = level
            elif policy == "dither":
                stock = np.maximum(level * (1.0 + cfg.dither_sd * rng.standard_normal(cfg.n_sku)), 0.0)
            elif policy == "explore":
                sd = np.sqrt(np.sum(w * (grid[None, :] - level[:, None]) ** 2, axis=1))
                stock = level + cfg.explore_z * sd
            else:
                raise ValueError(policy)

            p = SIZE / (SIZE + mu_true)
            demand = nbinom.rvs(SIZE, p, size=(DAYS, cfg.n_sku)).T.astype(float)
            sold = np.minimum(demand, stock[:, None])
            lost = demand - sold
            cum_lost += lost.sum()
            units += sold.sum()
            cum_cost += cfg.shortage * lost.sum() + cfg.holding * sold.sum()
            w = _posterior_update(w, sold, stock, grid)

            mean_post = np.sum(w * grid[None, :], axis=1)
            errs.append(float(np.mean(np.abs(mean_post - mu_true) / mu_true)))

        res["err"].append(errs[-1])
        res["cost"].append(cum_cost)
        res["fill"].append(units / max(units + cum_lost, 1e-9))
        res["inv"].append(float(np.mean(stock)))
        res["curve"] = errs
    for k in ("err", "cost", "fill", "inv"):
        res[k] = float(np.mean(res[k]))
    return res


if __name__ == "__main__":
    print(f"{'q':>5} {'policy':8} {'post err':>9} {'fill':>7} {'mean inv':>9} {'cost':>9}")
    for q in (0.5, 0.75, 0.91):
        for pol in ("greedy", "dither", "explore"):
            r = run(q=q, policy=pol)
            print(f"{q:5.2f} {pol:8} {100*r['err']:8.1f}% {r['fill']:7.3f} "
                  f"{r['inv']:9.2f} {r['cost']:9.0f}")
