"""Phase 4 tests — CAPM cost of equity & WACC (offline, pure math)."""
from __future__ import annotations

import pytest

from valuation.wacc import compute_wacc, cost_of_equity


def test_capm():
    # Rf 7% + beta 1.2 * ERP 6% = 14.2%
    assert cost_of_equity(0.07, 1.2, 0.06) == pytest.approx(0.142)


def test_wacc_weights_and_value():
    # E=800, D=200 -> wE=0.8, wD=0.2
    res = compute_wacc(
        equity_value=800, debt_value=200,
        beta=1.0, risk_free_rate=0.07, equity_risk_premium=0.06,
        cost_of_debt=0.09, tax_rate=0.25,
    )
    assert res.weight_equity == pytest.approx(0.8)
    assert res.weight_debt == pytest.approx(0.2)
    ke = 0.07 + 1.0 * 0.06            # 0.13
    kd_at = 0.09 * (1 - 0.25)         # 0.0675
    expected = 0.8 * ke + 0.2 * kd_at
    assert res.wacc == pytest.approx(expected)


def test_wacc_default_cost_of_debt():
    # cost_of_debt omitted -> Rf + 1.5% spread
    res = compute_wacc(equity_value=1000, debt_value=0,
                       risk_free_rate=0.07, equity_risk_premium=0.06, tax_rate=0.25)
    assert res.cost_of_debt_pretax == pytest.approx(0.085)
    # no debt -> WACC == cost of equity
    assert res.wacc == pytest.approx(res.cost_of_equity)


def test_wacc_override():
    res = compute_wacc(equity_value=1000, debt_value=500, wacc_override=0.11)
    assert res.wacc == 0.11


def test_equity_value_must_be_positive():
    with pytest.raises(ValueError):
        compute_wacc(equity_value=0, debt_value=100)
