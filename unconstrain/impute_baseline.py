"""A strong imputation baseline, standing in for the paper's stage-one imputer.

The FreshRetailNet baseline recovers latent demand with deep imputers (TimesNet,
SAITS, ImputeFormer, ...) and then forecasts on the recovered series.  Running
that stack needs Python 3.8, torch and a vendored PyPOTS; what matters for the
argument is its *structure*: interpolate the hole, then correct it with a model
trained on the observed days.  This module reproduces exactly that structure
with a gradient-boosted corrector, so the comparison isolates the idea
(impute-and-forecast) rather than a particular architecture.

Crucially, an imputer -- however deep -- is trained to reproduce *observed*
values.  It has no term for "the truth is above the observation", so it cannot
represent the censoring inequality.  That is the structural difference this
baseline is here to expose.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def _context(y: np.ndarray) -> np.ndarray:
    """Symmetric neighbourhood of every day: 3 before, 3 after, weekly lags."""
    n = y.size
    out = np.zeros((n, 8))
    for k, off in enumerate((-1, -2, -3, 1, 2, 3)):
        shifted = np.full(n, np.nan)
        if off > 0:
            shifted[:-off] = y[off:]
        else:
            shifted[-off:] = y[:off]
        out[:, k] = shifted
    out[:, 6] = y
    out[:, 7] = np.arange(n) % 7
    return out


def to_impute_gbm(observed, mask, exog=None, severity=None, **kw):
    """Interpolate, then let a model trained on clean days smooth the fill.

    This is the strongest form of the imputation approach that can be built
    from observed data alone: it sees both sides of the gap (a bidirectional
    imputer) and corrects the linear fill with a learned map.
    """
    from .methods import to_interp

    obs = np.asarray(observed, float)
    msk = np.asarray(mask, bool)
    base = to_interp(obs, msk)
    n, T = obs.shape

    xs, ys = [], []
    for i in range(n):
        clean = ~msk[i]
        if clean.sum() < 8:
            continue
        ctx = _context(base[i])
        xs.append(ctx[clean])
        ys.append(obs[i][clean])
        extra = [] if exog is None else [np.asarray(exog[i], float)[clean]]
        if extra:
            xs[-1] = np.concatenate([xs[-1], extra[0]], axis=1)
    if not xs:
        return base
    width = min(x.shape[1] for x in xs)
    X = np.concatenate([x[:, :width] for x in xs])
    y = np.concatenate(ys)
    model = HistGradientBoostingRegressor(
        max_iter=80, learning_rate=0.08, max_leaf_nodes=15,
        min_samples_leaf=20, early_stopping=False, random_state=0, **kw)
    model.fit(X, y)

    out = base.copy()
    for i in range(n):
        if not msk[i].any():
            continue
        ctx = _context(base[i])[msk[i]]
        if exog is not None:
            ctx = np.concatenate([ctx, np.asarray(exog[i], float)[msk[i]]], axis=1)
        out[i, msk[i]] = np.maximum(model.predict(ctx[:, :width]), 0.0)
    return out
