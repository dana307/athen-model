"""
Phase 15 — Stochastic discount rates (Vasicek) applied to the DCF.

Instead of one fixed WACC, the short rate follows a mean-reverting Vasicek
process:

    dr = κ(θ − r) dt + σ dW

Each simulated rate path implies a different discount-factor curve, so the FCFF
stream prices out to a *distribution* of intrinsic value driven purely by
interest-rate uncertainty. The discount rate each year is the short rate plus a
fixed risk spread.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.logging import get_logger

log = get_logger("actuarial.rates")


def simulate_vasicek(r0, kappa, theta, sigma, *, years, n_paths,
                     steps_per_year=1, seed=42):
    """Return simulated annual short-rate levels, shape (n_paths, years)."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / steps_per_year
    n_steps = years * steps_per_year
    r = np.full(n_paths, float(r0))
    annual = np.empty((n_paths, years))
    yr = 0
    for step in range(1, n_steps + 1):
        dr = kappa * (theta - r) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_paths)
        r = r + dr
        if step % steps_per_year == 0:
            annual[:, yr] = r
            yr += 1
    return annual


@dataclass
class StochasticDCFResult:
    prices: np.ndarray
    mean: float
    median: float
    std: float
    p5: float
    p95: float
    current_price: float | None = None
    prob_undervalued: float = float("nan")

    def summary(self) -> str:
        s = (f"mean ₹{self.mean:,.0f} / median ₹{self.median:,.0f} "
             f"(σ ₹{self.std:,.0f}) | 90% CI ₹{self.p5:,.0f}–₹{self.p95:,.0f}")
        if self.current_price is not None:
            s += f" | P(undervalued) {self.prob_undervalued:.0%}"
        return s


def stochastic_dcf(
    fcff,
    *,
    net_debt: float,
    shares_outstanding: float,
    r0: float = 0.07,
    kappa: float = 0.30,
    theta: float = 0.07,
    sigma: float = 0.015,
    spread: float = 0.05,
    terminal_growth: float = 0.045,
    n_paths: int = 10_000,
    seed: int = 42,
    current_price: float | None = None,
) -> StochasticDCFResult:
    """Discount an FCFF stream under Vasicek short-rate paths."""
    fcff = np.asarray(fcff, dtype=float)
    n = len(fcff)
    rates = simulate_vasicek(r0, kappa, theta, sigma, years=n,
                             n_paths=n_paths, seed=seed)
    disc_rates = rates + spread                      # (n_paths, n)

    # cumulative discount factors per path
    growth_factors = np.cumprod(1.0 + disc_rates, axis=1)
    df = 1.0 / growth_factors                        # (n_paths, n)

    pv_explicit = (fcff[None, :] * df).sum(axis=1)

    # terminal value uses the long-run mean discount rate (θ + spread)
    term_rate = theta + spread
    if term_rate <= terminal_growth:
        term_rate = terminal_growth + 0.01
    tv = fcff[-1] * (1 + terminal_growth) / (term_rate - terminal_growth)
    pv_tv = tv * df[:, -1]

    ev = pv_explicit + pv_tv
    equity = ev - net_debt
    prices = equity / shares_outstanding

    prob = float(np.mean(prices > current_price)) if current_price else float("nan")
    res = StochasticDCFResult(
        prices=prices, mean=float(prices.mean()), median=float(np.median(prices)),
        std=float(prices.std()), p5=float(np.percentile(prices, 5)),
        p95=float(np.percentile(prices, 95)),
        current_price=current_price, prob_undervalued=prob,
    )
    log.info("Stochastic DCF | %s", res.summary())
    return res
