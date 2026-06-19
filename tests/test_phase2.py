"""Phase 2 tests — derived metrics & driver profiling (offline)."""
from __future__ import annotations

import numpy as np
import pytest

from loaders.yfinance_loader import YFinanceLoader
from utils.metrics import add_derived_metrics, historical_drivers, DERIVED_FIELDS
from tests import fixtures


class FakeYFLoader(YFinanceLoader):
    def fetch_raw(self):
        return fixtures.raw_bundle()


@pytest.fixture
def fundamentals():
    return FakeYFLoader("RELIANCE").load()


def test_derived_columns_present(fundamentals):
    m = add_derived_metrics(fundamentals)
    for col in DERIVED_FIELDS:
        assert col in m.columns


def test_margin_math(fundamentals):
    m = add_derived_metrics(fundamentals)
    # 2024: EBITDA 1,60,000 / Revenue 9,00,000 = 0.1778
    assert m.loc[2024, "ebitda_margin"] == pytest.approx(1_60_000 / 9_00_000, rel=1e-6)


def test_net_debt(fundamentals):
    m = add_derived_metrics(fundamentals)
    # debt 3,20,000 - cash 2,00,000 = 1,20,000 cr
    assert m.loc[2024, "net_debt"] == pytest.approx(1_20_000e7)


def test_fcff_formula(fundamentals):
    m = add_derived_metrics(fundamentals, tax_rate=0.25).sort_index()
    row = m.loc[2024]
    nopat = 1_10_000e7 * 0.75
    expected = nopat + 50_000e7 - 1_20_000e7 - row["change_in_wc"]
    assert row["fcff"] == pytest.approx(expected)


def test_driver_profile(fundamentals):
    p = historical_drivers(fundamentals)
    assert p.latest_year == 2024
    assert p.latest_revenue == pytest.approx(9_00_000e7)
    assert p.n_years == 4
    # revenue grew 5,00,000 -> 9,00,000 over 3 yrs
    expected_cagr = (9_00_000 / 5_00_000) ** (1 / 3) - 1
    assert p.revenue_cagr == pytest.approx(expected_cagr, rel=1e-6)
    assert 0 < p.avg_ebitda_margin < 1


def test_profile_requires_two_years():
    import pandas as pd
    from utils.schema import CANONICAL_FIELDS
    one = pd.DataFrame(
        [[1.0] * len(CANONICAL_FIELDS)],
        index=[2024], columns=CANONICAL_FIELDS,
    )
    with pytest.raises(ValueError):
        historical_drivers(one)
