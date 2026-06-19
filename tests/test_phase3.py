"""Phase 3 tests — forecasting engine & methods (offline)."""
from __future__ import annotations

import numpy as np
import pytest

from loaders.yfinance_loader import YFinanceLoader
from forecasting import forecast, ForecastConfig, available
from forecasting.engine import FORECAST_COLUMNS
from utils.metrics import historical_drivers
from tests import fixtures


class FakeYFLoader(YFinanceLoader):
    def fetch_raw(self):
        return fixtures.raw_bundle()


@pytest.fixture
def fundamentals():
    return FakeYFLoader("RELIANCE").load()


def test_forecast_shape(fundamentals):
    proj = forecast(fundamentals, ForecastConfig(years=5))
    assert len(proj) == 5
    assert list(proj.columns) == FORECAST_COLUMNS
    assert list(proj.index) == [2025, 2026, 2027, 2028, 2029]


def test_cagr_revenue_compounds(fundamentals):
    p = historical_drivers(fundamentals)
    proj = forecast(fundamentals, ForecastConfig(years=3, revenue_method="cagr"))
    # year 1 revenue == latest * (1 + cagr)
    expected = p.latest_revenue * (1 + p.revenue_cagr)
    assert proj.iloc[0]["revenue"] == pytest.approx(expected, rel=1e-6)


def test_constant_growth_method(fundamentals):
    p = historical_drivers(fundamentals)
    proj = forecast(
        fundamentals,
        ForecastConfig(years=4, revenue_method="constant_growth",
                       revenue_params={"growth": 0.12}),
    )
    assert proj.iloc[0]["revenue"] == pytest.approx(p.latest_revenue * 1.12)
    assert proj.iloc[1]["revenue"] == pytest.approx(p.latest_revenue * 1.12 ** 2)


def test_fade_growth_decelerates(fundamentals):
    proj = forecast(fundamentals,
                    ForecastConfig(years=5, revenue_method="fade_growth"))
    g = proj["revenue_growth"].values
    # growth should be monotonically non-increasing as it fades to terminal
    assert all(g[i] >= g[i + 1] - 1e-9 for i in range(len(g) - 1))


def test_manual_revenue(fundamentals):
    vals = [1e13, 1.1e13, 1.2e13]
    proj = forecast(
        fundamentals,
        ForecastConfig(years=3, revenue_method="manual",
                       revenue_params={"values": vals}),
    )
    assert list(proj["revenue"].values) == pytest.approx(vals)


def test_ebit_identity(fundamentals):
    """EBIT must equal EBITDA - depreciation in every forecast year."""
    proj = forecast(fundamentals, ForecastConfig(years=5))
    assert np.allclose(proj["ebit"], proj["ebitda"] - proj["depreciation"])


def test_fcff_identity(fundamentals):
    """FCFF = NOPAT + D&A - capex - change in WC."""
    proj = forecast(fundamentals, ForecastConfig(years=5))
    rebuilt = (proj["nopat"] + proj["depreciation"]
               - proj["capex"] - proj["change_in_wc"])
    assert np.allclose(proj["fcff"], rebuilt)


def test_constant_margin_applied(fundamentals):
    proj = forecast(
        fundamentals,
        ForecastConfig(years=3, margin_method="constant",
                       margin_params={"value": 0.20}),
    )
    assert np.allclose(proj["ebitda_margin"], 0.20)
    assert np.allclose(proj["ebitda"], proj["revenue"] * 0.20)


def test_unknown_method_raises(fundamentals):
    with pytest.raises(KeyError):
        forecast(fundamentals, ForecastConfig(revenue_method="does_not_exist"))


def test_registries_populated():
    assert "cagr" in available("revenue")
    assert "regression" in available("margin")
    assert "percent_of_sales" in available("capex")
