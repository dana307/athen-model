"""
Revenue forecasting methods.

Each returns absolute projected revenue levels for `years` future periods.
Registered names: cagr, linear, constant_growth, fade_growth, manual.
"""
from __future__ import annotations

import numpy as np

from forecasting.base import register, fade


@register("revenue", "cagr")
def cagr(profile, history, years, growth=None, **_):
    """Grow the latest revenue at a constant rate (default: historical CAGR)."""
    g = profile.revenue_cagr if growth is None else growth
    base = profile.latest_revenue
    return base * np.power(1 + g, np.arange(1, years + 1))


@register("revenue", "constant_growth")
def constant_growth(profile, history, years, growth=0.10, **_):
    """Grow at a user-specified constant rate."""
    return cagr(profile, history, years, growth=growth)


@register("revenue", "fade_growth")
def fade_growth(profile, history, years, start=None, end=None, **_):
    """Glide the growth rate from `start` to `end` over the horizon.

    Defaults: start at the historical CAGR, fade to the terminal growth rate.
    This is the most realistic default for a maturing company.
    """
    from config import settings
    start = profile.revenue_cagr if start is None else start
    end = settings.TERMINAL_GROWTH if end is None else end
    rates = fade(start, end, years)
    levels, r = [], profile.latest_revenue
    for g in rates:
        r = r * (1 + g)
        levels.append(r)
    return np.array(levels, dtype=float)


@register("revenue", "linear")
def linear(profile, history, years, **_):
    """Extrapolate a straight line fit to historical revenue."""
    rev = history["revenue"].dropna().sort_index()
    x = np.arange(len(rev))
    slope, intercept = np.polyfit(x, rev.values, 1)
    future_x = np.arange(len(rev), len(rev) + years)
    return slope * future_x + intercept


@register("revenue", "manual")
def manual(profile, history, years, values=None, **_):
    """Use explicit revenue levels supplied by the analyst."""
    if values is None or len(values) != years:
        raise ValueError(f"manual revenue needs exactly {years} values")
    return np.asarray(values, dtype=float)
