"""unconstrain -- forecasting demand you never fully observe.

The core object of study is the censoring identity

    observed_sales = min(demand, available_stock)

A stock-out is therefore observationally identical to a demand collapse.  Any
forecaster trained on observed sales learns a systematically low demand level,
orders less, stocks out more often, and observes even lower sales.  The loop is
self-reinforcing and invisible in every metric computed on the observed series.

This package measures that loop and evaluates ways out of it.
"""

from .simulate import DemandConfig, Policy, simulate_demand, apply_censoring, simulate_inventory
from .methods import reconstruct, METHODS

__all__ = [
    "DemandConfig",
    "Policy",
    "simulate_demand",
    "apply_censoring",
    "simulate_inventory",
    "reconstruct",
    "METHODS",
]

__version__ = "0.1.0"
