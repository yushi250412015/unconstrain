"""Ways to reconstruct unobserved demand from censored sales.

Every method answers the same question: on a day when stock ran out, what was
demand *probably*?

    naive      use sales as if they were demand                 (the default)
    interp     linearly interpolate across stock-out days       (the common hack)
    mean       replace with the series mean of clean days       (the other hack)
    seasonal   replace with the same weekday's clean mean
    km         Kaplan-Meier de-censoring of the demand distribution
    tobit      Gaussian Tobit fitted by EM (interval-censored regression)
    nb         censored negative-binomial MLE with a log link

Partial censoring
-----------------
Retail stock-outs are rarely total.  In FreshRetailNet the median censored day
loses 29% of its selling hours, not all of them -- only 15% lose more than 60%.
So "demand exceeded sales" (the textbook ``d > c`` likelihood) is wrong for most
of the data: a day with one dead hour says almost nothing about unmet demand.

Each method therefore takes a *severity* ``s_t`` in [0, 1], the share of the
selling day that was unavailable, and uses the mixture likelihood

    (1 - s) * log p(y)  +  s * log P(D > y)

whose posterior mean is the convex combination ``(1-s) * y + s * E[D | D > y]``.
``s = 0`` is a clean day, ``s = 1`` is the textbook fully censored day.  The
``*_s`` variants in the registry use the severity; the plain ones force
``s = 1`` on any censored day, which is what the literature does by default.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import nbinom, norm

N_HARMONIC_WEEK = 2


# --------------------------------------------------------------------------
# design matrix
# --------------------------------------------------------------------------
def _design(n_days: int, exog_row: np.ndarray | None = None) -> np.ndarray:
    """Intercept + trend + seasonal shape (+ known covariates)."""
    t = np.arange(n_days, dtype=float)
    tc = (t - t.mean()) / max(t.std(), 1e-9)
    cols = [np.ones(n_days), tc]
    # a 90-day window barely spans a season: annual harmonics there are nearly
    # collinear with the trend and destabilise the per-series fit
    if n_days >= 180:
        cols += [np.sin(2 * np.pi * t / 365.0), np.cos(2 * np.pi * t / 365.0)]
    for k in range(1, N_HARMONIC_WEEK + 1):
        cols += [np.sin(2 * np.pi * k * t / 7.0), np.cos(2 * np.pi * k * t / 7.0)]
    X = np.stack(cols, axis=1)
    if exog_row is not None:
        X = np.concatenate([X, np.asarray(exog_row, float)], axis=1)
    return X


def _ridge_solve(X: np.ndarray, y: np.ndarray, lam: float = 1e-6) -> np.ndarray:
    k = X.shape[1]
    A = X.T @ X + lam * np.eye(k)
    A[0, 0] -= lam
    return np.linalg.solve(A, X.T @ y)


def _severity(mask: np.ndarray, severity: np.ndarray | None) -> np.ndarray:
    """Censoring severity, defaulting to the textbook all-or-nothing mask."""
    if severity is None:
        return np.asarray(mask, dtype=float)
    s = np.clip(np.asarray(severity, dtype=float), 0.0, 1.0)
    return np.where(mask, s, 0.0)


# --------------------------------------------------------------------------
# 1-4. the unmodelled approaches
# --------------------------------------------------------------------------
def to_naive(observed, mask, exog=None, severity=None):
    """Treat sales as demand.  The baseline every published pipeline uses."""
    return np.asarray(observed, dtype=float).copy()


def _interp_row(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return y.copy()
    idx = np.arange(len(y))
    good = ~mask
    if good.sum() < 2:
        fill = y[good].mean() if good.any() else 0.0
        return np.where(mask, fill, y)
    return np.interp(idx, idx[good], y[good])


def to_interp(observed, mask, exog=None, severity=None):
    obs = np.asarray(observed, float)
    msk = np.asarray(mask, bool)
    return np.stack([_interp_row(obs[i], msk[i]) for i in range(obs.shape[0])])


def to_mean(observed, mask, exog=None, severity=None):
    obs = np.asarray(observed, float)
    msk = np.asarray(mask, bool)
    out = obs.copy()
    for i in range(obs.shape[0]):
        clean = ~msk[i]
        if msk[i].any():
            fill = obs[i][clean].mean() if clean.any() else 0.0
            out[i, msk[i]] = fill
    return out


def to_seasonal(observed, mask, exog=None, severity=None):
    obs = np.asarray(observed, float)
    msk = np.asarray(mask, bool)
    out = obs.copy()
    weekday = np.arange(obs.shape[1]) % 7
    for i in range(obs.shape[0]):
        clean = ~msk[i]
        if not msk[i].any():
            continue
        fallback = obs[i][clean].mean() if clean.any() else 0.0
        for w in range(7):
            target = msk[i] & (weekday == w)
            if not target.any():
                continue
            pool = clean & (weekday == w)
            out[i, target] = obs[i][pool].mean() if pool.any() else fallback
    return out


# --------------------------------------------------------------------------
# 5. Kaplan-Meier
# --------------------------------------------------------------------------
def _km_survival(values: np.ndarray, censored: np.ndarray) -> np.ndarray:
    """Product-limit estimate S(j) = P(D > j) on integer support."""
    top = int(values.max()) if values.size else 0
    surv = np.ones(top + 2)
    s = 1.0
    for j in range(top + 1):
        at_risk = np.sum(values >= j)
        deaths = np.sum((values == j) & ~censored)
        if at_risk > 0 and deaths > 0:
            s *= 1.0 - deaths / at_risk
        surv[j] = s
    return surv


def _km_tail_mean(surv: np.ndarray, c: float) -> float:
    """E[D | D > c] = sum_j S(j) / S(c), the discrete survival area."""
    c = int(c)
    if c >= surv.size - 1:
        return float(c)
    sc = float(surv[c])
    if sc <= 1e-12:
        return float(c)
    return float(np.sum(surv[c:surv.size - 1]) / sc)


def to_km(observed, mask, exog=None, severity=None):
    """Kaplan-Meier tail mean, blended with the observation by severity."""
    obs = np.asarray(observed, float)
    msk = np.asarray(mask, bool)
    s = _severity(msk, severity)
    out = obs.copy()
    for i in range(obs.shape[0]):
        if not msk[i].any():
            continue
        surv = _km_survival(obs[i], msk[i])
        for t in np.nonzero(msk[i])[0]:
            tail = max(_km_tail_mean(surv, obs[i, t]), obs[i, t])
            out[i, t] = obs[i, t] + s[i, t] * (tail - obs[i, t])
    return out


# --------------------------------------------------------------------------
# 6. Gaussian Tobit by EM
# --------------------------------------------------------------------------
def _tobit_row(y, mask, X, s, n_iter=80, tol=1e-7):
    """EM for a severity-weighted censored normal.

    Complete-data log-likelihood  sum_t (1-s_t) log p(y_t) + s_t log P(D > y_t).
    The E-step pseudo-value is therefore the convex combination
    ``(1-s) y + s (mu + sigma lambda)``, and the M-step is plain least squares.
    """
    w = 1.0 - s
    if float(w.sum()) < X.shape[1] + 2:
        fill = float(np.sum(w * y) / max(w.sum(), 1e-9))
        return np.where(mask, fill, y)

    beta = _ridge_solve(X * np.sqrt(w)[:, None], y * np.sqrt(w))
    resid = y - X @ beta
    sigma = max(float(np.sqrt(np.sum(w * resid ** 2) / max(w.sum(), 1e-9))), 1e-3)

    for _ in range(n_iter):
        mu = X @ beta
        a = np.clip((y - mu) / sigma, -10.0, 10.0)
        lam = norm.pdf(a) / np.maximum(norm.sf(a), 1e-12)
        tail = mu + sigma * lam
        completed = w * y + s * tail
        beta_new = _ridge_solve(X, completed)
        mu_new = X @ beta_new

        var_tail = sigma ** 2 * np.maximum(1.0 + a * lam - lam ** 2, 1e-6)
        ss = np.sum(w * (y - mu_new) ** 2 + s * (var_tail + (tail - mu_new) ** 2))
        sigma_new = max(np.sqrt(ss / len(y)), 1e-3)

        delta = max(np.abs(beta_new - beta).max(), abs(sigma_new - sigma))
        beta, sigma = beta_new, sigma_new
        if delta < tol:
            break

    mu = X @ beta
    a = np.clip((y - mu) / sigma, -10.0, 10.0)
    lam = norm.pdf(a) / np.maximum(norm.sf(a), 1e-12)
    return np.maximum(w * y + s * (mu + sigma * lam), 0.0)


def to_tobit(observed, mask, exog=None, severity=None):
    obs = np.asarray(observed, float)
    msk = np.asarray(mask, bool)
    s = _severity(msk, severity)
    return np.stack([
        _tobit_row(obs[i], msk[i],
                   _design(obs.shape[1], None if exog is None else exog[i]), s[i])
        for i in range(obs.shape[0])
    ])


# --------------------------------------------------------------------------
# 7. censored negative binomial
# --------------------------------------------------------------------------
SIZE_GRID = (1.5, 5.0, 20.0, 60.0, 200.0, 2000.0)


def _nb_row(y, mask, X, s):
    """Severity-weighted censored NB2 MLE.

    ll = sum_t (1-s_t) log p(y_t)  +  sum_{mask} s_t log sf(y_t)
    """
    w = 1.0 - s
    if float(w.sum()) < X.shape[1] + 2:
        fill = float(np.sum(w * y) / max(w.sum(), 1e-9))
        return np.where(mask, fill, y)

    c_cen, X_cen, s_cen = y[mask], X[mask], s[mask]
    k = X.shape[1]

    def make_nll_grad(size):
        def nll_grad(beta):
            mu = np.exp(np.clip(X @ beta, -12.0, 12.0))
            ll_obs = np.sum(w * (gammaln(y + size) - gammaln(size) + size * np.log(size)
                                 - (size + y) * np.log(size + mu) + y * np.log(mu)))
            g = X.T @ (w * (mu * (size + y) / (size + mu) - y))
            if c_cen.size:
                mu_c = np.exp(np.clip(X_cen @ beta, -12.0, 12.0))
                p = size / (size + mu_c)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ls = nbinom.logsf(c_cen, size, p)
                    ls_prev = nbinom.logsf(c_cen - 1, size + 1.0, p)
                    tail = mu_c * np.exp(np.minimum(ls_prev - ls, 20.0))
                ls = np.nan_to_num(ls, nan=-1e6, neginf=-1e6)
                tail = np.nan_to_num(tail, nan=0.0, posinf=1e9)
                ll_obs = ll_obs + np.sum(s_cen * ls)
                g = g + X_cen.T @ (s_cen * (mu_c - tail))
            return -ll_obs, g
        return nll_grad

    beta0 = np.zeros(k)
    beta0[0] = np.log(max(float(np.sum(w * y) / max(w.sum(), 1e-9)), 0.1))

    best_f, best_beta, best_size = np.inf, None, SIZE_GRID[0]
    for size in SIZE_GRID:
        res = minimize(make_nll_grad(size), beta0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-7,
                                "maxls": 50})
        beta0 = res.x
        if res.fun < best_f:
            best_f, best_beta, best_size = res.fun, res.x, size

    w_mean = np.exp(np.clip(X @ best_beta, -12.0, 12.0))
    out = w * y + s * w_mean
    if c_cen.size:
        mu_c = w_mean[mask]
        p = best_size / (best_size + mu_c)
        with np.errstate(divide="ignore", invalid="ignore"):
            ls = nbinom.logsf(c_cen, best_size, p)
            ls_prev = nbinom.logsf(c_cen - 1, best_size + 1.0, p)
            tail = mu_c * np.exp(np.minimum(ls_prev - ls, 20.0))
        tail = np.maximum(np.nan_to_num(tail, nan=0.0, posinf=1e9), c_cen)
        out[mask] = w[mask] * y[mask] + s[mask] * tail
    return out


def to_nb(observed, mask, exog=None, severity=None):
    obs = np.asarray(observed, float)
    msk = np.asarray(mask, bool)
    s = _severity(msk, severity)
    return np.stack([
        _nb_row(obs[i], msk[i],
                _design(obs.shape[1], None if exog is None else exog[i]), s[i])
        for i in range(obs.shape[0])
    ])


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
def _force_hard(fn):
    """Ignore severity: treat every stock-out day as fully censored."""
    def wrapped(observed, mask, exog=None, severity=None):
        return fn(observed, mask, exog, None)
    wrapped.__name__ = fn.__name__ + "_hard"
    return wrapped


METHODS = {
    "naive": to_naive,
    "interp": to_interp,
    "mean": to_mean,
    "seasonal": to_seasonal,
    "km": _force_hard(to_km),
    "tobit": _force_hard(to_tobit),
    "nb": _force_hard(to_nb),
    "km_s": to_km,
    "tobit_s": to_tobit,
    "nb_s": to_nb,
}

METHOD_ORDER = ["naive", "interp", "mean", "seasonal",
                "km", "tobit", "nb", "km_s", "tobit_s", "nb_s"]

#: the subset that makes sense when only a binary mask exists
HARD_ORDER = ["naive", "interp", "mean", "seasonal", "km", "tobit", "nb"]

HACKS = {"naive", "interp", "mean", "seasonal"}
SOFT = {"km_s", "tobit_s", "nb_s"}

LABELS = {
    "naive": "naive (sales = demand)",
    "interp": "interpolate",
    "mean": "fill with mean",
    "seasonal": "fill with weekday mean",
    "km": "Kaplan-Meier",
    "tobit": "Tobit (Gaussian EM)",
    "nb": "censored neg-binomial",
    "km_s": "KM + severity",
    "tobit_s": "Tobit + severity",
    "nb_s": "neg-binomial + severity",
    "impute": "interp + learned correction",
}


def reconstruct(observed, mask, method: str, exog=None, severity=None) -> np.ndarray:
    """Reconstruct a demand panel from censored sales under ``method``."""
    if method == "impute":
        # imported lazily: it reuses to_interp from this module
        from .impute_baseline import to_impute_gbm
        out = to_impute_gbm(observed, mask, exog, severity)
        return np.nan_to_num(np.asarray(out, float), nan=0.0, posinf=0.0, neginf=0.0)
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
    out = np.asarray(METHODS[method](observed, mask, exog, severity), dtype=float)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def inflation(observed, mask, reconstructed) -> float:
    """How much a method forces total volume up, in percent."""
    s = float(np.asarray(observed, float).sum())
    if s <= 0:
        return float("nan")
    return float(100.0 * (np.asarray(reconstructed, float).sum() - s) / s)
