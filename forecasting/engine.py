"""
Forecast engine — assembles the projected operating model.

Given canonical fundamentals and a ForecastConfig, it walks each driver method
and builds a year-by-year projection from revenue all the way down to FCFF:

    revenue
      x margin            -> EBITDA
      - depreciation      -> EBIT
      x (1 - tax)         -> NOPAT
      + depreciation
      - capex
      - change in WC      -> FCFF   (the input to the Phase 5 DCF)

The result is a tidy DataFrame indexed by future fiscal year. Nothing here
knows *how* any driver was forecast — that's the registry's job — so swapping
methods never touches this file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# importing the method modules triggers their @register decorators
from forecasting import revenue as _revenue   # noqa: F401
from forecasting import margins as _margins    # noqa: F401
from forecasting import capex as _capex        # noqa: F401
from forecasting.base import ForecastConfig, get_method
from utils.metrics import add_derived_metrics, historical_drivers, DriverProfile
from utils.logging import get_logger

log = get_logger("forecasting")

FORECAST_COLUMNS = [
    "revenue", "revenue_growth", "ebitda_margin", "ebitda",
    "depreciation", "ebit", "capex", "working_capital",
    "change_in_wc", "nopat", "fcff",
]


class ForecastEngine:
    def __init__(self, config: ForecastConfig | None = None):
        self.config = config or ForecastConfig()

    def run(self, fundamentals: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        history = add_derived_metrics(fundamentals).sort_index(ascending=True)
        profile: DriverProfile = historical_drivers(fundamentals)
        n = cfg.years

        # --- resolve each driver via the registry -------------------------
        rev = get_method("revenue", cfg.revenue_method)(
            profile, history, n, **cfg.revenue_params)
        margin = get_method("margin", cfg.margin_method)(
            profile, history, n, **cfg.margin_params)
        capex_r = get_method("capex", cfg.capex_method)(
            profile, history, n, **cfg.capex_params)
        dep_r = get_method("dep", cfg.dep_method)(
            profile, history, n, **cfg.dep_params)
        wc_r = get_method("wc", cfg.wc_method)(
            profile, history, n, **cfg.wc_params)

        rev = np.asarray(rev, dtype=float)

        # --- assemble the operating model ---------------------------------
        ebitda = rev * margin
        depreciation = rev * dep_r
        ebit = ebitda - depreciation
        capex = rev * capex_r
        working_capital = rev * wc_r
        nopat = ebit * (1 - cfg.tax_rate)

        # change in working capital: first forecast year vs last actual
        last_wc = float(history["working_capital"].dropna().iloc[-1])
        wc_prev = np.concatenate([[last_wc], working_capital[:-1]])
        change_in_wc = working_capital - wc_prev

        fcff = nopat + depreciation - capex - change_in_wc

        # revenue growth vs prior year (first vs last actual revenue)
        last_rev = profile.latest_revenue
        rev_prev = np.concatenate([[last_rev], rev[:-1]])
        rev_growth = rev / rev_prev - 1

        years = [profile.latest_year + i for i in range(1, n + 1)]
        out = pd.DataFrame(
            {
                "revenue": rev,
                "revenue_growth": rev_growth,
                "ebitda_margin": margin,
                "ebitda": ebitda,
                "depreciation": depreciation,
                "ebit": ebit,
                "capex": capex,
                "working_capital": working_capital,
                "change_in_wc": change_in_wc,
                "nopat": nopat,
                "fcff": fcff,
            },
            index=pd.Index(years, name="fiscal_year"),
        )[FORECAST_COLUMNS]

        log.info(
            "forecast %d yrs | rev %s: %.0f -> %.0f cr | EBITDA margin ~%.1f%%",
            n, cfg.revenue_method, last_rev / 1e7, rev[-1] / 1e7,
            float(np.mean(margin)) * 100,
        )
        return out


def forecast(fundamentals: pd.DataFrame,
             config: ForecastConfig | None = None) -> pd.DataFrame:
    """Convenience wrapper."""
    return ForecastEngine(config).run(fundamentals)
