"""
Phase 1 tests — loader normalization, schema contract, and DB round-trip.

Run with:  pytest -q
These run fully offline using a yfinance-shaped fixture.
"""
from __future__ import annotations

import pandas as pd
import pytest

from loaders.yfinance_loader import YFinanceLoader
from loaders.base import LoaderError
from utils import database
from utils.schema import CANONICAL_FIELDS
from tests import fixtures


# --- a loader that uses the fixture instead of the network -----------------
class FakeYFLoader(YFinanceLoader):
    def fetch_raw(self):
        return fixtures.raw_bundle()


@pytest.fixture
def fundamentals():
    return FakeYFLoader("RELIANCE").load()


# --- schema contract --------------------------------------------------------
def test_columns_are_exactly_canonical(fundamentals):
    assert list(fundamentals.columns) == CANONICAL_FIELDS


def test_years_descending(fundamentals):
    assert list(fundamentals.index) == [2024, 2023, 2022, 2021]


def test_all_numeric(fundamentals):
    assert fundamentals.dtypes.apply(
        lambda d: pd.api.types.is_numeric_dtype(d)
    ).all()


# --- field-level correctness -----------------------------------------------
def test_revenue_mapped(fundamentals):
    assert fundamentals.loc[2024, "revenue"] == 9_00_000e7


def test_capex_made_positive(fundamentals):
    # fixture had -1,20,000 cr; loader must store it positive
    assert fundamentals.loc[2024, "capex"] == 1_20_000e7


def test_shares_outstanding_present(fundamentals):
    assert fundamentals.loc[2024, "shares_outstanding"] == 676e7


def test_working_capital_value(fundamentals):
    # WC reported neither directly: CA - CL = 2,50,000 - 3,00,000 = -50,000 cr
    assert fundamentals.loc[2024, "working_capital"] == (2_50_000e7 - 3_00_000e7)


# --- EBITDA/EBIT derivation -------------------------------------------------
def test_ebitda_derivation_when_missing():
    bundle = fixtures.raw_bundle()
    bundle["income"] = bundle["income"].drop(index="EBITDA")  # remove EBITDA row

    class L(YFinanceLoader):
        def fetch_raw(self):
            return bundle

    df = L("RELIANCE").load()
    # EBITDA should be reconstructed as EBIT + D&A
    expected = 1_10_000e7 + 50_000e7
    assert df.loc[2024, "ebitda"] == expected


# --- validation guards ------------------------------------------------------
def test_empty_data_raises():
    class Empty(YFinanceLoader):
        def fetch_raw(self):
            return {"income": pd.DataFrame(), "balance": pd.DataFrame(),
                    "cashflow": pd.DataFrame(), "info": {}}

    with pytest.raises(LoaderError):
        Empty("RELIANCE").load()


# --- SQLite round-trip ------------------------------------------------------
def test_db_roundtrip(fundamentals, tmp_path):
    db = tmp_path / "test.sqlite"
    n = database.save_fundamentals(fundamentals, "RELIANCE", "yfinance", db_path=db)
    assert n > 0

    back = database.load_fundamentals("RELIANCE", "yfinance", db_path=db)
    assert list(back.columns) == CANONICAL_FIELDS
    # values survive the round-trip
    assert back.loc[2024, "revenue"] == fundamentals.loc[2024, "revenue"]


def test_db_upsert_no_duplicates(fundamentals, tmp_path):
    db = tmp_path / "test.sqlite"
    database.save_fundamentals(fundamentals, "RELIANCE", "yfinance", db_path=db)
    database.save_fundamentals(fundamentals, "RELIANCE", "yfinance", db_path=db)  # again
    back = database.load_fundamentals("RELIANCE", "yfinance", db_path=db)
    # re-saving must refresh, not duplicate rows
    assert list(back.index) == [2024, 2023, 2022, 2021]
