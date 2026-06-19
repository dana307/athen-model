"""
Forecasting package (Phase 3).

    from forecasting import forecast, ForecastConfig
    proj = forecast(fundamentals, ForecastConfig(years=5, revenue_method="fade_growth"))
"""
from __future__ import annotations

from forecasting.base import (
    ForecastConfig,
    available,
    get_method,
    register,
)
from forecasting.engine import ForecastEngine, forecast, FORECAST_COLUMNS

__all__ = [
    "ForecastConfig",
    "ForecastEngine",
    "forecast",
    "FORECAST_COLUMNS",
    "available",
    "get_method",
    "register",
]
