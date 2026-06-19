"""
EBITDA-margin forecasting methods.

Each returns a projected EBITDA margin (fraction of revenue) per future year.
Registered names: constant, historical_average, latest, regression.
"""
from __future__ import annotations

import numpy as np

from forecasting.base import register


@register("margin", "constant")
def constant(profile, history, years, value=None, **_):
    """Hold a fixed margin (default: latest reported)."""
    m = profile.latest_ebitda_margin if value is None else value
    return np.full(years, m, dtype=float)


@register("margin", "latest")
def latest(profile, history, years, **_):
    return np.full(years, profile.latest_ebitda_margin, dtype=float)


@register("margin", "historical_average")
def historical_average(profile, history, years, **_):
    return np.full(years, profile.avg_ebitda_margin, dtype=float)


@register("margin", "regression")
def regression(profile, history, years, **_):
    """Linear-trend the historical EBITDA margin and project it forward.

    Clamped to [0, 1] so a steep trend can't produce an absurd margin.
    """
    s = history["ebitda_margin"].dropna().sort_index()
    if len(s) < 2:
        return historical_average(profile, history, years)
    x = np.arange(len(s))
    slope, intercept = np.polyfit(x, s.values, 1)
    future_x = np.arange(len(s), len(s) + years)
    proj = slope * future_x + intercept
    return np.clip(proj, 0.0, 1.0)
