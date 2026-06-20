"""
Phase 15 — Bayesian updating of assumptions.

Treat a valuation assumption (e.g. revenue growth) as a Normal prior and update
it with noisy observations using the Normal-Normal conjugate result. Posterior
precision is the sum of prior and data precisions:

    1/σ_post² = 1/σ_prior² + n/σ_obs²
    μ_post    = σ_post² · (μ_prior/σ_prior² + Σx_i/σ_obs²)

This lets Athena start from a prior view and tighten it as evidence arrives —
the actuarial habit of revising assumptions rather than fixing them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Posterior:
    mean: float
    std: float
    prior_mean: float
    prior_std: float
    n_obs: int

    def summary(self) -> str:
        return (f"prior {self.prior_mean:.2%}±{self.prior_std:.2%} → "
                f"posterior {self.mean:.2%}±{self.std:.2%} "
                f"({self.n_obs} obs)")


def normal_update(prior_mean, prior_std, observations, obs_std) -> Posterior:
    """Normal-Normal conjugate update with known observation noise."""
    obs = np.atleast_1d(np.asarray(observations, dtype=float))
    n = len(obs)
    prior_prec = 1.0 / prior_std ** 2
    obs_prec = n / obs_std ** 2
    post_var = 1.0 / (prior_prec + obs_prec)
    post_mean = post_var * (prior_mean * prior_prec + obs.sum() / obs_std ** 2)
    return Posterior(mean=float(post_mean), std=float(np.sqrt(post_var)),
                     prior_mean=float(prior_mean), prior_std=float(prior_std),
                     n_obs=int(n))


def update_growth_belief(prior_mean, prior_std, realized_growths, obs_std=0.03):
    """Convenience wrapper: update a growth assumption from realised history."""
    return normal_update(prior_mean, prior_std, realized_growths, obs_std)
