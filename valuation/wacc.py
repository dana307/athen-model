"""
Phase 4 — Cost of Capital (CAPM / WACC).

Computes the weighted-average cost of capital used to discount FCFF in Phase 5:

    Cost of equity (CAPM):  Ke = Rf + beta * ERP
    After-tax cost of debt: Kd_at = Kd * (1 - tax)
    WACC = (E/V) * Ke + (D/V) * Kd_at

Weights use *market* values: equity = price x shares, debt = book value of
total debt (a standard proxy for the market value of debt). Every intermediate
is returned in WaccResult so the assumption set is fully auditable in a report.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from config import settings
from utils.logging import get_logger

log = get_logger("valuation.wacc")


@dataclass
class WaccResult:
    # inputs
    risk_free_rate: float
    beta: float
    equity_risk_premium: float
    cost_of_debt_pretax: float
    tax_rate: float
    equity_value: float
    debt_value: float
    # derived
    cost_of_equity: float
    cost_of_debt_aftertax: float
    weight_equity: float
    weight_debt: float
    wacc: float

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"Ke={self.cost_of_equity:.2%} (Rf {self.risk_free_rate:.2%} + "
            f"β {self.beta:.2f} × ERP {self.equity_risk_premium:.2%}) | "
            f"Kd(at)={self.cost_of_debt_aftertax:.2%} | "
            f"wE={self.weight_equity:.0%} wD={self.weight_debt:.0%} | "
            f"WACC={self.wacc:.2%}"
        )


def cost_of_equity(risk_free_rate: float, beta: float,
                   equity_risk_premium: float) -> float:
    """CAPM."""
    return risk_free_rate + beta * equity_risk_premium


def compute_wacc(
    equity_value: float,
    debt_value: float,
    *,
    beta: float = 1.0,
    risk_free_rate: float | None = None,
    equity_risk_premium: float | None = None,
    cost_of_debt: float | None = None,
    tax_rate: float | None = None,
    wacc_override: float | None = None,
) -> WaccResult:
    """Compute WACC with market-value weights.

    Any unset assumption falls back to config defaults. `wacc_override` lets a
    user bypass CAPM entirely while still recording the inputs.
    """
    rf = settings.RISK_FREE_RATE if risk_free_rate is None else risk_free_rate
    erp = settings.EQUITY_RISK_PREMIUM if equity_risk_premium is None else equity_risk_premium
    tax = settings.DEFAULT_TAX_RATE if tax_rate is None else tax_rate
    # default cost of debt: risk-free + a modest credit spread
    kd = (rf + 0.015) if cost_of_debt is None else cost_of_debt

    if equity_value <= 0:
        raise ValueError("equity_value must be positive (price x shares).")
    if debt_value < 0:
        raise ValueError("debt_value cannot be negative.")

    v = equity_value + debt_value
    w_e = equity_value / v
    w_d = debt_value / v

    ke = cost_of_equity(rf, beta, erp)
    kd_at = kd * (1 - tax)
    wacc = w_e * ke + w_d * kd_at if wacc_override is None else wacc_override

    res = WaccResult(
        risk_free_rate=rf, beta=beta, equity_risk_premium=erp,
        cost_of_debt_pretax=kd, tax_rate=tax,
        equity_value=equity_value, debt_value=debt_value,
        cost_of_equity=ke, cost_of_debt_aftertax=kd_at,
        weight_equity=w_e, weight_debt=w_d, wacc=wacc,
    )
    log.info("WACC | %s", res.summary())
    return res
