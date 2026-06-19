"""
Forecasting method registries.

Mirrors the loader design: each driver (revenue, margin, capex, depreciation,
working capital) is a *family of interchangeable methods*. A method is just a
registered function, so adding "analyst estimate" or a fancier model later
means writing one function and decorating it — no engine changes.

Method contract
---------------
Every method has the uniform signature:

    method(profile: DriverProfile,
           history: pd.DataFrame,   # derived-metrics frame, ascending years
           years: int,
           **params) -> np.ndarray  # length == years

- Revenue methods return absolute revenue levels.
- Margin/ratio methods return a fraction (e.g. 0.18 for an 18% EBITDA margin,
  or 0.06 for capex at 6% of sales).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import settings

# name -> callable, one dict per driver
REVENUE_METHODS: dict = {}
MARGIN_METHODS: dict = {}
CAPEX_METHODS: dict = {}
DEP_METHODS: dict = {}
WC_METHODS: dict = {}

_REGISTRIES = {
    "revenue": REVENUE_METHODS,
    "margin": MARGIN_METHODS,
    "capex": CAPEX_METHODS,
    "dep": DEP_METHODS,
    "wc": WC_METHODS,
}


def register(driver: str, name: str):
    """Decorator: register a method under a driver family."""
    reg = _REGISTRIES[driver]

    def deco(fn):
        reg[name] = fn
        return fn

    return deco


def get_method(driver: str, name: str):
    reg = _REGISTRIES[driver]
    if name not in reg:
        raise KeyError(
            f"Unknown {driver} method '{name}'. Available: {sorted(reg)}"
        )
    return reg[name]


def available(driver: str) -> list[str]:
    return sorted(_REGISTRIES[driver])


@dataclass
class ForecastConfig:
    """Declarative spec for one forecast run.

    Each driver names a method + its params. Defaults give a reasonable
    'history-anchored' forecast out of the box.
    """
    years: int = settings.FORECAST_YEARS
    tax_rate: float = settings.DEFAULT_TAX_RATE

    revenue_method: str = "cagr"
    revenue_params: dict = field(default_factory=dict)

    margin_method: str = "historical_average"
    margin_params: dict = field(default_factory=dict)

    capex_method: str = "percent_of_sales"
    capex_params: dict = field(default_factory=dict)

    dep_method: str = "percent_of_sales"
    dep_params: dict = field(default_factory=dict)

    wc_method: str = "percent_of_sales"
    wc_params: dict = field(default_factory=dict)


def fade(start: float, end: float, years: int) -> np.ndarray:
    """Linear glide from `start` to `end` over `years` (useful for growth fade)."""
    if years == 1:
        return np.array([end], dtype=float)
    return np.linspace(start, end, years, dtype=float)
