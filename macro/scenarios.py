"""
Phase 14 — Macro scenario library.

A scenario expresses a macro event as *additive shocks* to the valuation
drivers (revenue growth, EBITDA margin, risk-free rate, equity risk premium,
terminal growth). The same event hits sectors differently, so each scenario
carries a base (economy-wide) shock plus optional sector-specific shocks — an
oil shock helps Energy producers and hurts Consumer/Auto, INR depreciation
helps IT exporters, and so on.

These deltas are deliberately stylised and transparent (not estimated betas),
so the propagation is easy to read and to tweak.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MacroShock:
    """Additive deltas applied to the base valuation assumptions."""
    d_revenue_growth: float = 0.0      # demand effect
    d_ebitda_margin: float = 0.0       # cost/pricing effect (pp as fraction)
    d_risk_free: float = 0.0           # feeds WACC via cost of equity
    d_erp: float = 0.0                 # risk-on/off
    d_terminal_growth: float = 0.0     # long-run effect

    def __add__(self, o: "MacroShock") -> "MacroShock":
        return MacroShock(
            self.d_revenue_growth + o.d_revenue_growth,
            self.d_ebitda_margin + o.d_ebitda_margin,
            self.d_risk_free + o.d_risk_free,
            self.d_erp + o.d_erp,
            self.d_terminal_growth + o.d_terminal_growth,
        )


@dataclass(frozen=True)
class StressScenario:
    key: str
    name: str
    description: str
    base: MacroShock
    sector_overrides: dict           # sector -> MacroShock (additive on base)

    def shock_for(self, sector: str) -> MacroShock:
        """Effective shock for a sector = base + sector adjustment."""
        adj = self.sector_overrides.get((sector or "").title(), MacroShock())
        return self.base + adj


# --------------------------------------------------------------------------
# Scenario library. Sector keys are Title-cased (e.g. "Energy", "It").
# --------------------------------------------------------------------------
SCENARIOS: dict[str, StressScenario] = {
    "oil_shock": StressScenario(
        "oil_shock", "Oil price shock (+50%)",
        "Crude spikes: input costs rise, inflation ticks up; producers gain.",
        base=MacroShock(d_revenue_growth=-0.010, d_ebitda_margin=-0.010,
                        d_risk_free=0.005, d_erp=0.005),
        sector_overrides={
            "Energy": MacroShock(d_revenue_growth=0.020, d_ebitda_margin=0.040),
            "Auto": MacroShock(d_ebitda_margin=-0.020, d_revenue_growth=-0.010),
            "Consumer": MacroShock(d_ebitda_margin=-0.015, d_revenue_growth=-0.010),
        },
    ),
    "rate_hike": StressScenario(
        "rate_hike", "Interest-rate hike (+200bps)",
        "Policy tightening lifts the discount rate and cools demand.",
        base=MacroShock(d_revenue_growth=-0.005, d_risk_free=0.020, d_erp=0.005),
        sector_overrides={
            "Financials": MacroShock(d_ebitda_margin=0.010, d_revenue_growth=-0.010),
            "Realty": MacroShock(d_revenue_growth=-0.020, d_ebitda_margin=-0.010),
            "Auto": MacroShock(d_revenue_growth=-0.015),
        },
    ),
    "inflation_spike": StressScenario(
        "inflation_spike", "Inflation spike",
        "Costs and nominal revenue both rise; margins compress, rates climb.",
        base=MacroShock(d_revenue_growth=0.010, d_ebitda_margin=-0.020,
                        d_risk_free=0.015, d_erp=0.010, d_terminal_growth=0.005),
        sector_overrides={
            "Consumer": MacroShock(d_ebitda_margin=-0.010),
            "Energy": MacroShock(d_ebitda_margin=0.010),
        },
    ),
    "inr_depreciation": StressScenario(
        "inr_depreciation", "INR depreciation (-10%)",
        "Weaker rupee helps exporters, raises import costs and imported inflation.",
        base=MacroShock(d_ebitda_margin=-0.005, d_risk_free=0.005),
        sector_overrides={
            "It": MacroShock(d_revenue_growth=0.020, d_ebitda_margin=0.020),
            "Pharma": MacroShock(d_revenue_growth=0.015, d_ebitda_margin=0.010),
            "Energy": MacroShock(d_ebitda_margin=-0.015),
            "Consumer": MacroShock(d_ebitda_margin=-0.010),
        },
    ),
    "recession": StressScenario(
        "recession", "Recession",
        "Demand contracts, risk premia widen; central bank eases.",
        base=MacroShock(d_revenue_growth=-0.040, d_ebitda_margin=-0.020,
                        d_risk_free=-0.005, d_erp=0.015, d_terminal_growth=-0.005),
        sector_overrides={
            "Consumer": MacroShock(d_revenue_growth=0.025, d_ebitda_margin=0.015),  # staples defensive
            "Pharma": MacroShock(d_revenue_growth=0.020, d_ebitda_margin=0.010),
            "Financials": MacroShock(d_ebitda_margin=-0.010),
        },
    ),
}


def available_scenarios() -> list[str]:
    return list(SCENARIOS)
