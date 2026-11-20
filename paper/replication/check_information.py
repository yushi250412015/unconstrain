"""Check the numerical information against a finite-difference likelihood score."""
import numpy as np
from scipy.stats import nbinom
from analysis import information
from unconstrain.identify import fisher_censored, retention

def outcome_probabilities(mu, k, cap):
    j = int(np.ceil(cap))
    p = k/(k+mu)
    return np.r_[nbinom.pmf(np.arange(j), k, p), nbinom.sf(j-1, k, p)]

for mu in (1.,5.,20.):
    for k in (10.,60.):
        for cap in (0.,.4,1.,2.,2.4,8.1):
            h = 1e-5*mu
            prob = outcome_probabilities(mu,k,cap)
            derivative = (outcome_probabilities(mu+h,k,cap)-outcome_probabilities(mu-h,k,cap))/(2*h)
            np.testing.assert_allclose(fisher_censored(cap,mu,k),sum(derivative**2/prob),atol=1e-10,rtol=1e-7)
        for s in (0.,.13,.32,.81,1.):
            np.testing.assert_allclose(information(mu,k,np.array([s]))[0],retention(mu,k,s),atol=1e-8)
print("Matched finite-difference information in 36 cap settings and retention in 30 severity settings.")

