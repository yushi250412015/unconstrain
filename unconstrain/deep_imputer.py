"""Deep imputers, as used by the FreshRetailNet baseline (SAITS via PyPOTS).

Requires the optional stack, none of which the rest of the package needs:

    pip install --no-deps pypots pygrinder benchpots ai4ts h5py tensorboard

The point of having this here is fairness.  My earlier stand-in (interp +
learned correction) is a *structural* analogue; this module runs the real
thing, so the comparison against the censored likelihood is measured rather
than argued.  It is slow: about 5 minutes on CPU for 1500 series x 90 days.
"""

from __future__ import annotations

import numpy as np


def to_saits(observed, mask, exog=None, severity=None, epochs: int = 100,
             d_model: int = 64, seed: int = 0, **_) -> np.ndarray:
    """Reconstruct censored days with SAITS, treating them as missing values.

    SAITS is trained to reproduce observed values under a reconstruction loss,
    so it inherits the distribution of the days it can see.  With half the days
    missing it leans on the visible level and cannot represent "the truth is
    above this observation".
    """
    import torch
    from pypots.imputation import SAITS

    torch.manual_seed(seed)
    np.random.seed(seed)

    obs = np.asarray(observed, float)
    msk = np.asarray(mask, bool)
    X = obs.copy()
    X[msk] = np.nan
    X3 = X[:, :, None]

    net = SAITS(n_steps=obs.shape[1], n_features=1, n_layers=2,
                d_model=d_model, n_heads=4, d_k=d_model // 2, d_v=d_model // 2,
                d_ffn=64, epochs=epochs, batch_size=128, patience=10,
                device="cpu", saving_path=None, verbose=False)
    net.fit({"X": X3})
    rec = np.asarray(net.impute({"X": X3}), dtype=float).reshape(obs.shape)
    return np.clip(np.nan_to_num(rec, nan=0.0), 0.0, None)
