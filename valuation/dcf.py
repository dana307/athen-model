"""
Phase 5 — FCFF Discounted Cash Flow engine.

Takes the Phase 3 forecast's FCFF stream and the Phase 4 WACC and produces an
intrinsic per-share value:

    PV(explicit FCFF)  = Σ FCFF_t / (1+WACC)^t
    Terminal value     = FCFF_n * (1+g) / (WACC - g)      [Gordon growth]
    PV(terminal value) = TV / (1+WACC)^n
    Enterprise value   = PV(explicit) + PV(terminal)
    Equity value       = EV - net debt
    Intrinsic price     = Equity value / shares outstanding

An optional mid-year convention discounts explicit flows at t-0.5 (cash arrives
through the year, not only at year-end).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

from config import settings
from utils.logging import get_logger

log = get_logger("valuation.dcf")


@dataclass
class DCFResult:
    wacc: float
    terminal_growth: float
    pv_explicit: float
    terminal_value: float
    pv_terminal: float
    enterprise_value: float
    net_debt: float
    equity_value: float
    shares_outstanding: float
    intrinsic_price: float
    current_price: float | None = None
    upside: float | None = None
    pv_by_year: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        tv_share = self.pv_terminal / self.enterprise_value if self.enterprise_value else float("nan")
        s = (
            f"EV={self.enterprise_value/1e7:,.0f} cr "
            f"(TV {tv_share:.0%}) | "
            f"Equity={self.equity_value/1e7:,.0f} cr | "
            f"Intrinsic=₹{self.intrinsic_price:,.0f}/sh"
        )
        if self.current_price is not None:
            s += f" | Mkt ₹{self.current_price:,.0f} | Upside {self.upside:+.1%}"
        return s


def run_dcf(
    forecast_df: pd.DataFrame,
    wacc: float,
    *,
    net_debt: float,
    shares_outstanding: float,
    terminal_growth: float | None = None,
    current_price: float | None = None,
    mid_year: bool = False,
) -> DCFResult:
    g = settings.TERMINAL_GROWTH if terminal_growth is None else terminal_growth

    # --- guards (the assertions the roadmap calls for) ---------------------
    if shares_outstanding <= 0:
        raise ValueError("shares_outstanding must be > 0")
    if wacc <= g:
        raise ValueError(
            f"WACC ({wacc:.2%}) must exceed terminal growth ({g:.2%}); "
            "Gordon-growth terminal value is otherwise invalid/negative."
        )

    fcff = forecast_df["fcff"].to_numpy(dtype=float)
    n = len(fcff)
    if n == 0:
        raise ValueError("forecast has no FCFF rows")

    periods = np.arange(1, n + 1, dtype=float)
    disc_periods = periods - 0.5 if mid_year else periods
    disc_factors = 1.0 / np.power(1.0 + wacc, disc_periods)

    pv_fcff = fcff * disc_factors
    pv_explicit = float(pv_fcff.sum())

    # terminal value sits at end of year n; discount at full n periods
    terminal_value = fcff[-1] * (1 + g) / (wacc - g)
    pv_terminal = float(terminal_value / np.power(1.0 + wacc, n))

    enterprise_value = pv_explicit + pv_terminal
    equity_value = enterprise_value - net_debt
    intrinsic_price = equity_value / shares_outstanding

    upside = None
    if current_price:
        upside = intrinsic_price / current_price - 1

    res = DCFResult(
        wacc=wacc, terminal_growth=g,
        pv_explicit=pv_explicit, terminal_value=float(terminal_value),
        pv_terminal=pv_terminal, enterprise_value=enterprise_value,
        net_debt=net_debt, equity_value=equity_value,
        shares_outstanding=shares_outstanding, intrinsic_price=intrinsic_price,
        current_price=current_price, upside=upside,
        pv_by_year=[float(x) for x in pv_fcff],
    )
    log.info("DCF | %s", res.summary())
    return res
