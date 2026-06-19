"""
Intensity-driver methods: capex, depreciation, and working capital.

All three are modelled as a fraction of sales (the standard operating-model
convention), so they share the same method shapes. Each returns a *ratio* per
future year; the engine multiplies by projected revenue to get absolute levels.

Registered names per driver: percent_of_sales, user_defined.
(Working capital additionally supports `constant_ratio`.)
"""
from __future__ import annotations

import numpy as np

from forecasting.base import register


# ---- capex ----------------------------------------------------------------
@register("capex", "percent_of_sales")
def capex_pct_sales(profile, history, years, pct=None, **_):
    p = profile.avg_capex_pct_sales if pct is None else pct
    return np.full(years, p, dtype=float)


@register("capex", "user_defined")
def capex_user(profile, history, years, pct=None, values=None, **_):
    return _user_defined(years, pct, values)


# ---- depreciation ---------------------------------------------------------
@register("dep", "percent_of_sales")
def dep_pct_sales(profile, history, years, pct=None, **_):
    p = profile.avg_dep_pct_sales if pct is None else pct
    return np.full(years, p, dtype=float)


@register("dep", "user_defined")
def dep_user(profile, history, years, pct=None, values=None, **_):
    return _user_defined(years, pct, values)


# ---- working capital ------------------------------------------------------
@register("wc", "percent_of_sales")
def wc_pct_sales(profile, history, years, pct=None, **_):
    p = profile.avg_wc_pct_sales if pct is None else pct
    return np.full(years, p, dtype=float)


@register("wc", "user_defined")
def wc_user(profile, history, years, pct=None, values=None, **_):
    return _user_defined(years, pct, values)


# ---- shared helper --------------------------------------------------------
def _user_defined(years, pct, values):
    """Either a single ratio held flat, or an explicit per-year ratio array."""
    if values is not None:
        if len(values) != years:
            raise ValueError(f"user_defined needs exactly {years} values")
        return np.asarray(values, dtype=float)
    if pct is None:
        raise ValueError("user_defined requires `pct` or `values`")
    return np.full(years, pct, dtype=float)
