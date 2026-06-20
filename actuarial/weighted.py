"""
Phase 15 — Scenario-weighted (probability-weighted) valuation.

Takes the Phase 14 macro stress results plus a probability for each scenario
(and the base case) and collapses them into a single expected intrinsic value,
its standard deviation, and the probability the stock is undervalued. This is
the scenario-weighted-valuation piece of the actuarial layer: value isn't one
number, it's an expectation over a probability-weighted set of futures.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WeightedValuation:
    expected_value: float
    std: float
    probabilities: dict
    values: dict
    market_price: float | None = None
    prob_undervalued: float = float("nan")

    def summary(self) -> str:
        s = f"E[intrinsic] = ₹{self.expected_value:,.0f} (σ ₹{self.std:,.0f})"
        if self.market_price is not None:
            s += f" | P(undervalued) {self.prob_undervalued:.0%}"
        return s


def scenario_weighted_value(stress_result, probabilities: dict,
                            *, include_base: bool = True) -> WeightedValuation:
    """Probability-weight scenario intrinsic values into an expectation.

    `probabilities` maps scenario key (and optionally 'base') to a weight; the
    weights are normalised. Missing scenarios get zero weight.
    """
    values: dict[str, float] = {}
    if include_base:
        values["base"] = stress_result.base_intrinsic
    for key, s in stress_result.scenarios.items():
        values[key] = s["intrinsic"]

    # normalise the supplied probabilities over the keys we actually have
    probs = {k: float(probabilities.get(k, 0.0)) for k in values}
    total = sum(probs.values())
    if total <= 0:
        raise ValueError("probabilities must sum to a positive number")
    probs = {k: v / total for k, v in probs.items()}

    keys = list(values)
    v = np.array([values[k] for k in keys])
    p = np.array([probs[k] for k in keys])
    ev = float((v * p).sum())
    var = float((p * (v - ev) ** 2).sum())
    std = float(np.sqrt(var))

    mp = stress_result.market_price
    prob_under = float(p[v > mp].sum()) if mp else float("nan")

    return WeightedValuation(expected_value=ev, std=std, probabilities=probs,
                             values=values, market_price=mp,
                             prob_undervalued=prob_under)
