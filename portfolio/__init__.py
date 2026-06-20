"""
Portfolio package (Phases 12-13).

    from portfolio import Portfolio, Holding, build_holding
    p = Portfolio([h1, h2, h3])
    analytics = p.analyze()
"""
from __future__ import annotations

from portfolio.analytics import (
    Portfolio, Holding, PortfolioAnalytics, build_holding,
)
from portfolio.optimizer import (
    min_variance, max_sharpe, mean_variance, risk_parity,
    efficient_frontier, black_litterman, portfolio_performance,
    returns_from_prices, annualized_inputs, OptimizationResult,
)

__all__ = [
    "Portfolio", "Holding", "PortfolioAnalytics", "build_holding",
    "min_variance", "max_sharpe", "mean_variance", "risk_parity",
    "efficient_frontier", "black_litterman", "portfolio_performance",
    "returns_from_prices", "annualized_inputs", "OptimizationResult",
]
