"""Phase 14 tests — macro stress testing (offline)."""
from __future__ import annotations

import pytest

from loaders.yfinance_loader import YFinanceLoader
from macro import stress_test, available_scenarios, SCENARIOS
from macro.scenarios import StressScenario, MacroShock
from tests import fixtures


def _load(bundle):
    class L(YFinanceLoader):
        def fetch_raw(self):
            return bundle
    return L("X").load()


@pytest.fixture
def healthy():
    return _load(fixtures.raw_bundle())


def test_all_scenarios_run(healthy):
    res = stress_test(healthy, market_price=2900, ticker="RELIANCE")
    assert set(res.scenarios) == set(available_scenarios())
    assert res.base_intrinsic > 0


def test_zero_shock_equals_base(healthy):
    """A null scenario must reproduce the base intrinsic exactly."""
    null = {"flat": StressScenario("flat", "Flat", "no shock",
                                   base=MacroShock(), sector_overrides={})}
    res = stress_test(healthy, market_price=2900, scenarios=null)
    assert res.scenarios["flat"]["intrinsic"] == pytest.approx(res.base_intrinsic)
    assert res.scenarios["flat"]["change"] == pytest.approx(0.0, abs=1e-9)


def test_recession_reduces_value(healthy):
    res = stress_test(healthy, market_price=2900, sector="default")
    assert res.scenarios["recession"]["change"] < 0


def test_rate_hike_raises_wacc_and_cuts_value(healthy):
    res = stress_test(healthy, market_price=2900, sector="default")
    # higher rates -> higher discount rate -> lower intrinsic
    assert res.scenarios["rate_hike"]["change"] < 0
    assert res.scenarios["rate_hike"]["rf"] > 0.07   # base rf shocked up


def test_sector_differentiation_oil_shock(healthy):
    """Energy should fare better than Consumer under an oil shock."""
    energy = stress_test(healthy, market_price=2900, sector="Energy")
    consumer = stress_test(healthy, market_price=2900, sector="Consumer")
    assert (energy.scenarios["oil_shock"]["change"]
            > consumer.scenarios["oil_shock"]["change"])


def test_it_benefits_from_inr_depreciation(healthy):
    it = stress_test(healthy, market_price=2900, sector="IT")
    default = stress_test(healthy, market_price=2900, sector="default")
    assert (it.scenarios["inr_depreciation"]["change"]
            > default.scenarios["inr_depreciation"]["change"])


def test_table_shape(healthy):
    res = stress_test(healthy, market_price=2900)
    assert len(res.table) == len(SCENARIOS)
    assert "change_%" in res.table.columns
