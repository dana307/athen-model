"""
Phase 14 — Macro stress engine.

Takes a company, applies each macro scenario's shocks to the base valuation
assumptions, recomputes the WACC (the risk-free-rate and ERP shocks flow
through CAPM), re-prices via the DCF, and reports the intrinsic value and the
% impact versus the unshocked base case.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config import settings
from utils.metrics import historical_drivers
from valuation import intrinsic_price
from valuation.wacc import compute_wacc
from macro.scenarios import SCENARIOS, StressScenario, MacroShock
from utils.logging import get_logger

log = get_logger("macro.stress")


@dataclass
class StressResult:
    ticker: str
    sector: str
    base_intrinsic: float
    market_price: float
    scenarios: dict                      # key -> {name, intrinsic, change, ...}
    table: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        worst = min(self.scenarios.values(), key=lambda s: s["intrinsic"])
        return (f"base ₹{self.base_intrinsic:,.0f} | worst: {worst['name']} "
                f"₹{worst['intrinsic']:,.0f} ({worst['change']:+.0%})")


def _price_under(fundamentals, *, growth, margin, rf, erp, tg, beta,
                 market_price, shares, debt, tax, years):
    wacc = compute_wacc(
        equity_value=market_price * shares, debt_value=debt,
        beta=beta, risk_free_rate=rf, equity_risk_premium=erp, tax_rate=tax,
    ).wacc
    # keep the Gordon constraint satisfied
    tg = min(tg, wacc - 0.005)
    return intrinsic_price(fundamentals, growth=growth, margin=margin,
                           wacc=wacc, terminal_growth=tg, years=years, tax_rate=tax)


def stress_test(
    fundamentals: pd.DataFrame,
    *,
    market_price: float,
    ticker: str = "",
    sector: str = "default",
    beta: float = 1.0,
    risk_free_rate: float | None = None,
    equity_risk_premium: float | None = None,
    terminal_growth: float | None = None,
    years: int = settings.FORECAST_YEARS,
    scenarios: dict[str, StressScenario] | None = None,
) -> StressResult:
    scenarios = scenarios or SCENARIOS
    rf0 = settings.RISK_FREE_RATE if risk_free_rate is None else risk_free_rate
    erp0 = settings.EQUITY_RISK_PREMIUM if equity_risk_premium is None else equity_risk_premium
    tg0 = settings.TERMINAL_GROWTH if terminal_growth is None else terminal_growth
    tax = settings.DEFAULT_TAX_RATE

    profile = historical_drivers(fundamentals)
    g0, m0 = profile.revenue_cagr, profile.avg_ebitda_margin
    shares = float(fundamentals.sort_index(ascending=False)["shares_outstanding"].dropna().iloc[0])
    debt = float(fundamentals.sort_index(ascending=False)["debt"].dropna().iloc[0])

    # base (unshocked) intrinsic — same pathway, zero deltas
    base = _price_under(fundamentals, growth=g0, margin=m0, rf=rf0, erp=erp0,
                        tg=tg0, beta=beta, market_price=market_price,
                        shares=shares, debt=debt, tax=tax, years=years)

    rows, results = [], {}
    for key, scen in scenarios.items():
        s: MacroShock = scen.shock_for(sector)
        g = g0 + s.d_revenue_growth
        m = max(m0 + s.d_ebitda_margin, 0.01)
        rf = max(rf0 + s.d_risk_free, 0.0)
        erp = max(erp0 + s.d_erp, 0.0)
        tg = tg0 + s.d_terminal_growth
        price = _price_under(fundamentals, growth=g, margin=m, rf=rf, erp=erp,
                             tg=tg, beta=beta, market_price=market_price,
                             shares=shares, debt=debt, tax=tax, years=years)
        change = price / base - 1 if base else float("nan")
        results[key] = {"name": scen.name, "intrinsic": price, "change": change,
                        "growth": g, "margin": m, "rf": rf, "erp": erp, "tg": tg}
        rows.append({"scenario": scen.name, "intrinsic": round(price, 1),
                     "change_%": round(change * 100, 1),
                     "growth_%": round(g * 100, 1), "margin_%": round(m * 100, 1),
                     "rf_%": round(rf * 100, 2), "erp_%": round(erp * 100, 2)})

    table = pd.DataFrame(rows).set_index("scenario")
    res = StressResult(ticker=ticker.upper(), sector=sector,
                       base_intrinsic=base, market_price=market_price,
                       scenarios=results, table=table)
    log.info("Stress [%s/%s] | %s", ticker.upper(), sector, res.summary())
    return res
