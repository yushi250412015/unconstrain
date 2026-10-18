"""How much information does a censored observation actually carry?

The de-censoring literature compares estimators. This module asks the prior
question: given the censoring the data actually has, how much information about
the demand level is *left*, and therefore how small can any estimator's error
possibly be?

For a day observed with threshold ``c`` (sales are capped at ``c``), the
observation is a mixture: ``D < c`` is seen exactly, ``D >= c`` collapses into a
single event. Its Fisher information about the level ``mu`` is

    I(c, mu) = sum_{y<c} p(y) (d/dmu log p(y))^2
             + P(D>=c) (d/dmu log P(D>=c))^2

and the ratio ``rho = I / I(c=inf)`` is the fraction of a clean day's
information that survives. ``n`` censored days are worth ``n * rho`` clean ones,
and the Cramer-Rao bound turns that into a floor on the standard error of any
unbiased level estimate. No imputer, transformer or foundation model can go
below it: they all see the same censored data.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import nbinom


def _p(mu: float, k: float) -> float:
    return k / (k + mu)


def logpmf(y, mu: float, k: float):
    return nbinom.logpmf(y, k, _p(mu, k))


def logsf(y, mu: float, k: float):
    """log P(D > y)."""
    return nbinom.logsf(y, k, _p(mu, k))


def _d(fn, y, mu, k, h=1e-5) -> np.ndarray:
    return (fn(y, mu + h, k) - fn(y, mu - h, k)) / (2 * h)


def fisher_clean(mu: float, k: float, y_max: int | None = None) -> float:
    """Fisher information of one uncensored observation."""
    y_max = y_max or int(mu + 12 * np.sqrt(mu + mu ** 2 / k) + 20)
    ys = np.arange(y_max + 1)
    p = nbinom.pmf(ys, k, _p(mu, k))
    d = _d(lambda yy, m, kk: logpmf(yy, m, kk), ys, mu, k)
    return float(np.sum(p * d ** 2))


def fisher_censored(c: float, mu: float, k: float) -> float:
    """Fisher information of one observation censored at ``c``."""
    ys = np.arange(0, int(c))
    if ys.size == 0:
        ys = np.array([], dtype=int)
    out = 0.0
    if ys.size:
        p = nbinom.pmf(ys, k, _p(mu, k))
        d = _d(lambda yy, m, kk: logpmf(yy, m, kk), ys, mu, k)
        out += float(np.sum(p * d ** 2))
    surv = nbinom.sf(c - 1, k, _p(mu, k))          # P(D >= c)
    dsf = _d(lambda yy, m, kk: logsf(yy - 1, m, kk), np.array([c]), mu, k)[0]
    out += float(surv * dsf ** 2)
    return out


def threshold_for_severity(mu: float, k: float, s: float) -> float:
    """Stock level at which a fraction ``s`` of demand volume is lost.

    Solves E[min(D, c)] = (1 - s) mu, i.e. the policy implied by the observed
    severity in the data.
    """
    if s <= 0:
        return np.inf
    lo, hi = 0.0, mu + 12 * np.sqrt(mu + mu ** 2 / k) + 20
    for _ in range(60):
        mid = (lo + hi) / 2
        ys = np.arange(0, int(mid) + 1)
        p = nbinom.pmf(ys, k, _p(mu, k))
        served = float(np.sum(ys * p)) + mid * float(nbinom.sf(int(mid), k, _p(mu, k)))
        if served / mu < 1 - s:
            lo = mid
        else:
            hi = mid
    return hi


def retention(mu: float, k: float, s: float) -> float:
    """Fraction of a clean observation's Fisher information kept at severity s."""
    if s <= 0:
        return 1.0
    c = threshold_for_severity(mu, k, s)
    return fisher_censored(c, mu, k) / fisher_clean(mu, k)


def se_bound(mu: float, k: float, s: float, n_days: int) -> float:
    """Cramer-Rao floor on the standard error of any unbiased estimate of mu.

    The bound is for a *scale* parameter, so it is reported as a relative error.
    """
    info = n_days * retention(mu, k, s) * fisher_clean(mu, k)
    return float(1.0 / np.sqrt(max(info, 1e-12)) / mu)


if __name__ == "__main__":
    k, n = 60.0, 90
    print(f"NB2(k={k:g}), n={n} days. rho = information retained by a censored day\n")
    print(f"{'severity':>9} {'rho':>7} {'eff. days':>10} {'SE bound':>10}")
    for s in (0.0, 0.05, 0.1, 0.2, 0.32, 0.5, 0.7, 0.9):
        mus = np.geomspace(0.5, 30, 40)
        r = float(np.mean([retention(m, k, s) for m in mus]))
        print(f"{s:9.2f} {r:7.3f} {r*n:10.1f} {100*se_bound(1.0, k, s, n):9.2f}%")
