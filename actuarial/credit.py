"""
Phase 15 — Structural credit model (Merton).

Merton (1974) treats a firm's equity as a call option on its assets struck at
the face value of debt. From observable equity value and equity volatility we
back out the unobservable asset value V and asset volatility σ_V by solving:

    E   = V·N(d1) − D·e^{−rT}·N(d2)
    σ_E·E = N(d1)·σ_V·V

then the distance to default DD = d2 and the (risk-neutral) default probability
PD = N(−d2). This is the actuarial bridge: equity-market data → a probability
of default, the same object a credit/risk team would price.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm

from utils.logging import get_logger

log = get_logger("actuarial.credit")


@dataclass
class CreditResult:
    asset_value: float
    asset_vol: float
    distance_to_default: float
    default_probability: float
    equity_value: float
    debt_face: float
    horizon: float

    def summary(self) -> str:
        return (f"PD({self.horizon:.0f}y)={self.default_probability:.2%} | "
                f"DD={self.distance_to_default:.2f} | "
                f"asset vol {self.asset_vol:.1%}")


def merton_pd(
    equity_value: float,
    equity_vol: float,
    debt_face: float,
    *,
    risk_free_rate: float = 0.07,
    horizon: float = 1.0,
) -> CreditResult:
    """Solve the Merton model for the default probability over `horizon` years."""
    if equity_value <= 0 or debt_face <= 0:
        raise ValueError("equity_value and debt_face must be positive")

    E, sigE, D, r, T = equity_value, equity_vol, debt_face, risk_free_rate, horizon

    def equations(x):
        V, sigV = x
        V = max(V, 1e-6)
        sigV = max(sigV, 1e-6)
        d1 = (np.log(V / D) + (r + 0.5 * sigV ** 2) * T) / (sigV * np.sqrt(T))
        d2 = d1 - sigV * np.sqrt(T)
        eq1 = V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E
        eq2 = norm.cdf(d1) * sigV * V - sigE * E
        return [eq1, eq2]

    # initial guess: assets ≈ equity + debt, asset vol scaled down by leverage
    V0 = E + D
    sigV0 = sigE * E / (E + D)
    V, sigV = fsolve(equations, [V0, sigV0], full_output=False)
    V, sigV = float(max(V, 1e-6)), float(max(sigV, 1e-6))

    d1 = (np.log(V / D) + (r + 0.5 * sigV ** 2) * T) / (sigV * np.sqrt(T))
    d2 = d1 - sigV * np.sqrt(T)
    dd = float(d2)
    pd = float(norm.cdf(-d2))

    res = CreditResult(asset_value=V, asset_vol=sigV, distance_to_default=dd,
                       default_probability=pd, equity_value=E, debt_face=D,
                       horizon=T)
    log.info("Merton | %s", res.summary())
    return res


def pd_from_leverage(net_debt: float, ebitda: float) -> float:
    """A crude reduced-form fallback: map net-debt/EBITDA to a 1y PD band.

    Used only when equity volatility for the Merton model is unavailable.
    """
    if ebitda <= 0:
        return 0.30
    lev = net_debt / ebitda
    # stylised, monotone mapping
    return float(np.clip(0.005 + 0.02 * max(lev, 0), 0.0, 0.5))
