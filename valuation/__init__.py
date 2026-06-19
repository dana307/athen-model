"""
Valuation package (Phases 4-5+).

    from valuation import value_company
    res = value_company(fundamentals, market_price=2900, beta=1.1)
    print(res.intrinsic_price)
"""
from __future__ import annotations

from valuation.wacc import compute_wacc, cost_of_equity, WaccResult
from valuation.dcf import run_dcf, DCFResult
from valuation.multiples import (
    compute_multiples, peer_comparison, CompanyMultiples, ComparablesResult,
)
from valuation.montecarlo import simulate, MonteCarloConfig, MonteCarloResult
from valuation.sensitivity import (
    wacc_growth_grid, grid_axes, save_heatmap,
)
from valuation.pipeline import value_company, ValuationResult, intrinsic_price

__all__ = [
    "compute_wacc", "cost_of_equity", "WaccResult",
    "run_dcf", "DCFResult",
    "compute_multiples", "peer_comparison", "CompanyMultiples", "ComparablesResult",
    "simulate", "MonteCarloConfig", "MonteCarloResult",
    "wacc_growth_grid", "grid_axes", "save_heatmap",
    "value_company", "ValuationResult", "intrinsic_price",
]
