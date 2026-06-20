"""
Actuarial layer (Phase 15, stretch).

Probabilistic modelling on top of the equity valuation:

    merton_pd            — structural default probability (distance to default)
    stochastic_dcf       — Vasicek stochastic-rate value distribution
    normal_update        — Bayesian updating of assumptions
    scenario_weighted_value — probability-weighted scenario valuation
"""
from __future__ import annotations

from actuarial.credit import merton_pd, pd_from_leverage, CreditResult
from actuarial.rates import stochastic_dcf, simulate_vasicek, StochasticDCFResult
from actuarial.bayes import normal_update, update_growth_belief, Posterior
from actuarial.weighted import scenario_weighted_value, WeightedValuation

__all__ = [
    "merton_pd", "pd_from_leverage", "CreditResult",
    "stochastic_dcf", "simulate_vasicek", "StochasticDCFResult",
    "normal_update", "update_growth_belief", "Posterior",
    "scenario_weighted_value", "WeightedValuation",
]
