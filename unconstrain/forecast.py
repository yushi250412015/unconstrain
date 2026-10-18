"""One forecaster, many histories.

The forecaster is deliberately ordinary: gradient-boosted trees, direct
multi-step, lag/rolling/calendar features.  The experiment varies only what it
is *trained on*, so any difference in outcome is attributable to the
de-censoring, not to model capacity.

``R_feat`` and ``R_target`` are separate on purpose.  A method can clean up the
labels it regresses onto while still feeding the model lag features built from
censored sales -- an ablation worth having, because it is the mistake people
actually make.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

LAGS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 21, 28)
WINDOWS = (7, 14, 28)
MAX_LAG = max(LAGS)


def _cum(A: np.ndarray) -> np.ndarray:
    """Cumulative sum with a leading zero column, for O(1) window sums."""
    return np.concatenate([np.zeros((A.shape[0], 1)), np.cumsum(A, axis=1)], axis=1)


def _feature_block(R: np.ndarray, mask: np.ndarray, origins: np.ndarray, h: int,
                   cs: np.ndarray, cm: np.ndarray, scale: np.ndarray,
                   sid: np.ndarray, promo_col: np.ndarray | None = None) -> np.ndarray:
    n, T = R.shape
    m = origins.size
    tg = origins + h
    feats = []

    for lag in LAGS:
        feats.append(R[:, origins - lag])
    for w in WINDOWS:
        feats.append((cs[:, origins + 1] - cs[:, origins + 1 - w]) / w)

    win = np.stack([R[:, origins - k] for k in range(1, 8)], axis=0)
    feats.append(win.std(axis=0))

    # calendar of the *target* day -- this is known at forecast time
    feats.append(np.broadcast_to(np.sin(2 * np.pi * tg / 7.0), (n, m)))
    feats.append(np.broadcast_to(np.cos(2 * np.pi * tg / 7.0), (n, m)))
    feats.append(np.broadcast_to(np.sin(2 * np.pi * tg / 365.0), (n, m)))
    feats.append(np.broadcast_to(np.cos(2 * np.pi * tg / 365.0), (n, m)))
    feats.append(np.broadcast_to(tg / 365.0, (n, m)))

    # observable censoring pressure: share of the last 7 days that were capped
    feats.append((cm[:, origins + 1] - cm[:, origins + 1 - 7]) / 7.0)

    feats.append(np.broadcast_to(scale, (n, m)))
    feats.append(np.broadcast_to(sid, (n, m)))
    if promo_col is not None:
        feats.append(np.broadcast_to(promo_col, (n, m)))

    return np.stack(feats, axis=-1).reshape(n * m, -1)


class DirectForecaster:
    """Direct h-step-ahead boosted trees, one model per horizon."""

    def __init__(self, horizons: int = 7, max_iter: int = 120,
                 learning_rate: float = 0.08, random_state: int = 0,
                 quantile: float | None = None):
        self.horizons = int(horizons)
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)
        #: None = conditional mean; a number = that quantile (pinball loss)
        self.quantile = None if quantile is None else float(quantile)
        self.models: dict[int, HistGradientBoostingRegressor] = {}
        self.residual_q: dict[int, tuple[float, float]] = {}

    # -- fit -------------------------------------------------------------
    def fit(self, R_feat: np.ndarray, R_target: np.ndarray, mask: np.ndarray,
            origins: np.ndarray, promo: np.ndarray | None = None) -> "DirectForecaster":
        R_feat = np.asarray(R_feat, float)
        R_target = np.asarray(R_target, float)
        mask = np.asarray(mask, bool)
        cs, cm = _cum(R_feat), _cum(mask.astype(float))
        scale = np.maximum(R_feat.mean(axis=1, keepdims=True), 1e-6)
        sid = (np.arange(R_feat.shape[0], dtype=float) / max(R_feat.shape[0], 1))[:, None]
        self._cs_shape = R_feat.shape

        for h in range(1, self.horizons + 1):
            promo_col = None if promo is None else np.asarray(promo)[:, origins + h]
            X = _feature_block(R_feat, mask, origins, h, cs, cm, scale, sid, promo_col)
            y = R_target[:, origins + h].reshape(-1)
            model = HistGradientBoostingRegressor(
                max_iter=self.max_iter, learning_rate=self.learning_rate,
                max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
                early_stopping=False, random_state=self.random_state,
                **({} if self.quantile is None else
                   {"loss": "quantile", "quantile": self.quantile}),
            )
            model.fit(X, y)
            self.models[h] = model
            resid = y - model.predict(X)
            lo, hi = np.quantile(resid, [0.10, 0.90])
            self.residual_q[h] = (float(lo), float(hi))
        return self

    # -- predict ---------------------------------------------------------
    def predict(self, R_feat: np.ndarray, mask: np.ndarray, origin: int,
                promo_future: np.ndarray | None = None) -> np.ndarray:
        """Forecast ``horizons`` days ahead of ``origin`` (inclusive index)."""
        R_feat = np.asarray(R_feat, float)
        mask = np.asarray(mask, bool)
        cs, cm = _cum(R_feat), _cum(mask.astype(float))
        scale = np.maximum(R_feat.mean(axis=1, keepdims=True), 1e-6)
        sid = (np.arange(R_feat.shape[0], dtype=float) / max(R_feat.shape[0], 1))[:, None]
        origins = np.array([int(origin)])

        out = np.zeros((R_feat.shape[0], self.horizons))
        for h in range(1, self.horizons + 1):
            promo_col = None if promo_future is None else promo_future[:, [h - 1]]
            X = _feature_block(R_feat, mask, origins, h, cs, cm, scale, sid, promo_col)
            out[:, h - 1] = np.maximum(self.models[h].predict(X), 0.0)
        return out

    def interval(self, forecast: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Training-residual interval, for the coverage monitor."""
        lo = np.zeros_like(forecast)
        hi = np.zeros_like(forecast)
        for h in range(1, self.horizons + 1):
            a, b = self.residual_q[h]
            lo[:, h - 1] = np.maximum(forecast[:, h - 1] + a, 0.0)
            hi[:, h - 1] = forecast[:, h - 1] + b
        return lo, hi


def fit_predict(R_feat, R_target, mask, train_end: int, origin: int,
                horizons: int = 7, **kw) -> np.ndarray:
    """Convenience: fit on ``[.., train_end)`` and forecast from ``origin``."""
    origins = np.arange(MAX_LAG + 1, train_end - horizons)
    model = DirectForecaster(horizons=horizons, **kw).fit(R_feat, R_target, mask, origins)
    return model.predict(R_feat, mask, origin)
