"""
Valuation pipeline — orchestrates Phases 3-5 into one intrinsic value.

    forecast (Phase 3) -> WACC (Phase 4) -> DCF (Phase 5)

This is the function reports and the dashboard will call. It keeps the wiring
(latest shares, net debt, market-value equity weight) in one place so callers
just supply fundamentals + a few market inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import settings
from forecasting import forecast as run_forecast, ForecastConfig
from utils.metrics import historical_drivers
from valuation.wacc import compute_wacc, WaccResult
from valuation.dcf import run_dcf, DCFResult


@dataclass
class ValuationResult:
    forecast: pd.DataFrame
    wacc: WaccResult
    dcf: DCFResult

    @property
    def intrinsic_price(self) -> float:
        return self.dcf.intrinsic_price


def _latest(fundamentals: pd.DataFrame, field: str) -> float:
    s = fundamentals.sort_index(ascending=False)[field].dropna()
    if s.empty:
        raise ValueError(f"no value for '{field}' in fundamentals")
    return float(s.iloc[0])


def value_company(
    fundamentals: pd.DataFrame,
    *,
    market_price: float,
    beta: float = 1.0,
    forecast_config: ForecastConfig | None = None,
    risk_free_rate: float | None = None,
    equity_risk_premium: float | None = None,
    cost_of_debt: float | None = None,
    tax_rate: float | None = None,
    terminal_growth: float | None = None,
    mid_year: bool = False,
) -> ValuationResult:
    cfg = forecast_config or ForecastConfig()
    if tax_rate is not None:
        cfg.tax_rate = tax_rate

    profile = historical_drivers(fundamentals)
    shares = _latest(fundamentals, "shares_outstanding")
    debt = _latest(fundamentals, "debt")
    net_debt = profile.latest_net_debt

    # 1) forecast the operating model -> FCFF stream
    proj = run_forecast(fundamentals, cfg)

    # 2) discount rate (market-value equity weight)
    equity_value = market_price * shares
    wacc_res = compute_wacc(
        equity_value=equity_value,
        debt_value=debt,
        beta=beta,
        risk_free_rate=risk_free_rate,
        equity_risk_premium=equity_risk_premium,
        cost_of_debt=cost_of_debt,
        tax_rate=cfg.tax_rate,
    )

    # 3) intrinsic value
    dcf_res = run_dcf(
        proj,
        wacc_res.wacc,
        net_debt=net_debt,
        shares_outstanding=shares,
        terminal_growth=terminal_growth,
        current_price=market_price,
        mid_year=mid_year,
    )

    return ValuationResult(forecast=proj, wacc=wacc_res, dcf=dcf_res)


def intrinsic_price(
    fundamentals: pd.DataFrame,
    *,
    growth: float,
    margin: float,
    wacc: float,
    terminal_growth: float | None = None,
    years: int = settings.FORECAST_YEARS,
    tax_rate: float | None = None,
) -> float:
    """Fast DCF for given point assumptions — powers the dashboard sliders.

    Uses constant growth + constant margin so the two headline drivers map
    directly to the sliders; capex/D&A/WC stay at historical averages.
    """
    tax = settings.DEFAULT_TAX_RATE if tax_rate is None else tax_rate
    tg = settings.TERMINAL_GROWTH if terminal_growth is None else terminal_growth
    tg = min(tg, wacc - 0.005)

    profile = historical_drivers(fundamentals)
    shares = _latest(fundamentals, "shares_outstanding")
    cfg = ForecastConfig(
        years=years, tax_rate=tax,
        revenue_method="constant_growth", revenue_params={"growth": growth},
        margin_method="constant", margin_params={"value": margin},
    )
    proj = run_forecast(fundamentals, cfg)
    return run_dcf(proj, wacc, net_debt=profile.latest_net_debt,
                   shares_outstanding=shares, terminal_growth=tg).intrinsic_price
