"""Phase 12 tests — portfolio analytics (offline)."""
from __future__ import annotations

import numpy as np
import pytest

from portfolio import Portfolio, Holding, build_holding
from loaders.yfinance_loader import YFinanceLoader
from tests import fixtures


def _p():
    # three holdings with known values
    return Portfolio([
        Holding("A", quantity=100, market_price=100, intrinsic_price=120, sector="IT"),
        Holding("B", quantity=50, market_price=200, intrinsic_price=180, sector="Energy"),
        Holding("C", quantity=200, market_price=50, intrinsic_price=75, sector="IT"),
    ])


def test_market_and_intrinsic_value():
    a = _p().analyze()
    # MV: 100*100 + 50*200 + 200*50 = 10000 + 10000 + 10000 = 30000
    assert a.total_market_value == pytest.approx(30000)
    # IV: 100*120 + 50*180 + 200*75 = 12000 + 9000 + 15000 = 36000
    assert a.total_intrinsic_value == pytest.approx(36000)
    assert a.portfolio_upside == pytest.approx(36000 / 30000 - 1)   # +20%


def test_weights_sum_to_one():
    a = _p().analyze()
    assert sum(a.weights.values()) == pytest.approx(1.0)
    # each holding is 10000/30000 here
    for w in a.weights.values():
        assert w == pytest.approx(1 / 3)


def test_sector_allocation():
    a = _p().analyze()
    # A and C are IT (2/3), B is Energy (1/3)
    assert a.sector_allocation["IT"] == pytest.approx(2 / 3)
    assert a.sector_allocation["Energy"] == pytest.approx(1 / 3)
    assert sum(a.sector_allocation.values()) == pytest.approx(1.0)


def test_concentration_hhi():
    a = _p().analyze()
    # equal thirds -> HHI = 3*(1/3)^2 = 1/3; effective holdings = 3
    assert a.hhi == pytest.approx(1 / 3)
    assert a.effective_holdings == pytest.approx(3.0)
    assert a.concentration_level == "Concentrated"   # HHI 0.33 > 0.25


def test_diversified_label():
    # 10 equal holdings -> HHI 0.1 -> Diversified
    hs = [Holding(f"T{i}", 1, 100, 110) for i in range(10)]
    a = Portfolio(hs).analyze()
    assert a.hhi == pytest.approx(0.1)
    assert a.concentration_level == "Diversified"


def test_weighted_expected_return_defaults_to_gap():
    a = _p().analyze()
    # default expected return == valuation gap; weighted == portfolio upside
    assert a.weighted_valuation_gap == pytest.approx(a.portfolio_upside)
    assert a.weighted_expected_return == pytest.approx(a.weighted_valuation_gap)


def test_explicit_expected_return():
    p = Portfolio([
        Holding("A", 100, 100, 120, expected_return=0.10),
        Holding("B", 100, 100, 120, expected_return=0.20),
    ])
    a = p.analyze()
    assert a.weighted_expected_return == pytest.approx(0.15)   # equal weights


def test_empty_portfolio_raises():
    with pytest.raises(ValueError):
        Portfolio().analyze()


def test_build_holding_from_fundamentals():
    class L(YFinanceLoader):
        def fetch_raw(self):
            return fixtures.raw_bundle()
    df = L("RELIANCE").load()
    h = build_holding("RELIANCE", df, market_price=2900, quantity=10,
                      sector="Energy", beta=1.1)
    assert h.ticker == "RELIANCE"
    assert h.intrinsic_price > 0
    assert h.market_value == pytest.approx(29000)
