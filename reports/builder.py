"""
Phase 10 — Research report builder.

Pulls together every Athena engine into a single ResearchReport object that the
DOCX/PDF renderers (and the dashboard) consume:

    fundamentals · forecast · WACC · DCF · comparables · Monte Carlo · risk
    + bull/base/bear scenarios + a blended fair value and BUY/HOLD/SELL call.

The builder owns the *analysis*; the renderers own the *formatting*. That split
means the same report content drives both the Word/PDF output and the dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from config import settings
from forecasting import forecast, ForecastConfig
from utils.metrics import historical_drivers, add_derived_metrics
from valuation import value_company, peer_comparison
from valuation.dcf import run_dcf
from valuation.montecarlo import simulate, MonteCarloConfig
from risk import assess, RiskLevel
from utils.logging import get_logger

log = get_logger("reports.builder")

# recommendation thresholds on blended upside
BUY_THRESHOLD = 0.15
SELL_THRESHOLD = -0.15


@dataclass
class ResearchReport:
    ticker: str
    company_name: str
    as_of: str
    currency: str
    market_price: float

    fundamentals: pd.DataFrame
    metrics: pd.DataFrame
    forecast: pd.DataFrame

    wacc: object
    dcf: object
    montecarlo: object
    risk: object
    comps: object | None

    scenarios: dict
    fair_value: float
    upside: float
    recommendation: str
    thesis: list = field(default_factory=list)
    key_risks: list = field(default_factory=list)
    charts: dict = field(default_factory=dict)


def _scenario_price(fundamentals, *, growth, margin, wacc, tg, tax,
                    years, net_debt, shares) -> float:
    cfg = ForecastConfig(
        years=years, tax_rate=tax,
        revenue_method="constant_growth", revenue_params={"growth": growth},
        margin_method="constant", margin_params={"value": margin},
    )
    proj = forecast(fundamentals, cfg)
    tg = min(tg, wacc - 0.005)
    return run_dcf(proj, wacc, net_debt=net_debt, shares_outstanding=shares,
                   terminal_growth=tg).intrinsic_price


def _recommendation(upside: float) -> str:
    if upside >= BUY_THRESHOLD:
        return "BUY"
    if upside <= SELL_THRESHOLD:
        return "REDUCE"
    return "HOLD"


def build_report(
    fundamentals: pd.DataFrame,
    *,
    ticker: str,
    market_price: float,
    company_name: str | None = None,
    beta: float = 1.0,
    years: int = settings.FORECAST_YEARS,
    risk_free_rate: float | None = None,
    equity_risk_premium: float | None = None,
    terminal_growth: float | None = None,
    peers: dict | None = None,
) -> ResearchReport:
    company_name = company_name or ticker.upper()
    profile = historical_drivers(fundamentals)
    metrics = add_derived_metrics(fundamentals)

    # --- base valuation (headline) ----------------------------------------
    val = value_company(
        fundamentals, market_price=market_price, beta=beta,
        forecast_config=ForecastConfig(years=years),
        risk_free_rate=risk_free_rate, equity_risk_premium=equity_risk_premium,
        terminal_growth=terminal_growth,
    )
    base_wacc = val.wacc.wacc
    base_tg = val.dcf.terminal_growth
    net_debt, shares = val.dcf.net_debt, val.dcf.shares_outstanding
    tax = val.wacc.tax_rate

    # --- Monte Carlo & risk -----------------------------------------------
    mc = simulate(
        fundamentals,
        MonteCarloConfig(n_sims=10_000, years=years,
                         wacc=(base_wacc, 0.01), terminal_growth=(base_tg, 0.005)),
        current_price=market_price,
    )
    risk = assess(fundamentals)

    # --- comparables (optional) -------------------------------------------
    comps = None
    if peers:
        comps = peer_comparison(fundamentals, ticker, market_price, peers)

    # --- bull / base / bear scenarios -------------------------------------
    g0, m0 = profile.revenue_cagr, profile.avg_ebitda_margin
    scen_defs = {
        "Bull": dict(growth=g0 + 0.02, margin=m0 + 0.01, wacc=base_wacc - 0.01, tg=base_tg + 0.005),
        "Base": dict(growth=g0, margin=m0, wacc=base_wacc, tg=base_tg),
        "Bear": dict(growth=g0 - 0.02, margin=max(m0 - 0.01, 0.01), wacc=base_wacc + 0.01, tg=base_tg - 0.005),
    }
    scenarios = {}
    for name, d in scen_defs.items():
        price = _scenario_price(fundamentals, tax=tax, years=years,
                                net_debt=net_debt, shares=shares, **d)
        scenarios[name] = {
            "price": price,
            "upside": price / market_price - 1,
            "assumptions": d,
        }

    # --- blended fair value + recommendation ------------------------------
    fair_components = [val.dcf.intrinsic_price]
    if comps is not None and not np.isnan(comps.implied_price):
        fair_components.append(comps.implied_price)
    fair_value = float(np.mean(fair_components))
    upside = fair_value / market_price - 1
    rec = _recommendation(upside)

    # --- narrative --------------------------------------------------------
    thesis = _build_thesis(profile, val, mc, risk, comps, upside)
    key_risks = [f"{f.label}: {f.note}" for f in risk.findings
                 if f.level >= RiskLevel.MEDIUM] or ["No material red flags detected."]

    report = ResearchReport(
        ticker=ticker.upper(), company_name=company_name,
        as_of=date.today().isoformat(), currency=settings.REPORTING_CURRENCY,
        market_price=market_price,
        fundamentals=fundamentals, metrics=metrics, forecast=val.forecast,
        wacc=val.wacc, dcf=val.dcf, montecarlo=mc, risk=risk, comps=comps,
        scenarios=scenarios, fair_value=fair_value, upside=upside,
        recommendation=rec, thesis=thesis, key_risks=key_risks,
    )
    log.info("Report | %s %s | fair ₹%.0f vs ₹%.0f (%.1f%%) -> %s",
             ticker.upper(), report.as_of, fair_value, market_price,
             upside * 100, rec)
    return report


def _build_thesis(profile, val, mc, risk, comps, upside) -> list[str]:
    pts = []
    pts.append(
        f"Revenue compounded ~{profile.revenue_cagr*100:.1f}% over the historical "
        f"window with an average EBITDA margin of {profile.avg_ebitda_margin*100:.1f}%."
    )
    pts.append(
        f"DCF intrinsic value is ₹{val.dcf.intrinsic_price:,.0f}/share at a "
        f"{val.wacc.wacc*100:.1f}% WACC and {val.dcf.terminal_growth*100:.1f}% "
        f"terminal growth ({val.dcf.upside:+.0%} vs market)."
    )
    if comps is not None and not np.isnan(comps.implied_price):
        pts.append(
            f"Peer multiples imply ₹{comps.implied_price:,.0f}/share "
            f"({comps.upside:+.0%}), a market-based cross-check on the DCF."
        )
    pts.append(
        f"Monte Carlo (10k sims) centres on ₹{mc.mean:,.0f}/share "
        f"(90% CI ₹{mc.p5:,.0f}–₹{mc.p95:,.0f}); P(undervalued) "
        f"= {mc.prob_undervalued:.0%}."
    )
    pts.append(f"Overall financial risk is rated {risk.overall_level}.")
    return pts
