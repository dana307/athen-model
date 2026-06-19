"""Phase 8 tests — sensitivity grid (offline)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from valuation.dcf import run_dcf
from valuation.sensitivity import wacc_growth_grid, grid_axes


def _flat_forecast(fcff=100.0, n=5):
    return pd.DataFrame({"fcff": [fcff] * n},
                        index=[2025 + i for i in range(n)])


def test_grid_axes_centered():
    waccs, growths = grid_axes(0.12, 0.045, n=2)
    assert len(waccs) == 5 and len(growths) == 5
    assert waccs[2] == pytest.approx(0.12)
    assert growths[2] == pytest.approx(0.045)


def test_grid_shape_and_labels():
    waccs, growths = grid_axes(0.12, 0.045, n=2)
    grid = wacc_growth_grid(_flat_forecast(), waccs, growths,
                            net_debt=50.0, shares_outstanding=10.0)
    assert grid.shape == (5, 5)
    assert list(grid.index) == list(growths)
    assert list(grid.columns) == list(waccs)


def test_invalid_cells_are_nan():
    # wacc 0.04 with growth 0.05 -> invalid (wacc <= g)
    grid = wacc_growth_grid(_flat_forecast(), [0.04, 0.10], [0.03, 0.05],
                            net_debt=0.0, shares_outstanding=10.0)
    assert np.isnan(grid.loc[0.05, 0.04])     # wacc <= g
    assert not np.isnan(grid.loc[0.03, 0.10])  # valid


def test_cells_match_run_dcf():
    fc = _flat_forecast()
    grid = wacc_growth_grid(fc, [0.10, 0.12], [0.03, 0.04],
                            net_debt=50.0, shares_outstanding=10.0)
    expected = run_dcf(fc, 0.12, net_debt=50.0, shares_outstanding=10.0,
                       terminal_growth=0.04).intrinsic_price
    assert grid.loc[0.04, 0.12] == pytest.approx(expected)


def test_monotonicity():
    """Higher WACC -> lower value; higher terminal growth -> higher value."""
    grid = wacc_growth_grid(_flat_forecast(), [0.10, 0.12, 0.14],
                            [0.02, 0.03, 0.04],
                            net_debt=0.0, shares_outstanding=10.0)
    # across a row (rising WACC) value falls
    row = grid.loc[0.03]
    assert row[0.10] > row[0.12] > row[0.14]
    # down a column (rising growth) value rises
    col = grid[0.12]
    assert col[0.02] < col[0.03] < col[0.04]
