"""
Loader abstraction.

A loader's only job: turn a ticker into a clean, schema-conforming
fundamentals DataFrame. Every concrete source (yfinance now, screener.in
later) subclasses BaseLoader and implements `fetch_raw` + `normalize`.

The template method `load()` ties them together and validates the result,
so all loaders are guaranteed to emit the same contract.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from utils.schema import CANONICAL_FIELDS, POSITIVE_REQUIRED
from utils.logging import get_logger

log = get_logger("loaders")


class LoaderError(RuntimeError):
    """Raised when a loader cannot produce valid fundamentals."""


class BaseLoader(ABC):
    """Base class for all data sources."""

    #: short identifier, e.g. "yfinance"
    source_name: str = "base"

    def __init__(self, ticker: str):
        self.ticker = ticker.strip().upper()

    # --- subclasses implement these two ------------------------------------
    @abstractmethod
    def fetch_raw(self) -> Any:
        """Return source-specific raw data (DataFrames, JSON, HTML, ...)."""

    @abstractmethod
    def normalize(self, raw: Any) -> pd.DataFrame:
        """Map raw data onto CANONICAL_FIELDS. Index = fiscal years (desc)."""

    def market_data(self) -> dict:
        """Optional live market inputs for valuation (price, beta, market_cap).

        Default: nothing. Sources that can provide it (yfinance) override this.
        """
        return {}

    # --- template method ----------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Fetch, normalize, validate. Returns the canonical fundamentals."""
        log.info("[%s] fetching %s", self.source_name, self.ticker)
        raw = self.fetch_raw()
        df = self.normalize(raw)
        df = self._coerce_schema(df)
        self._validate(df)
        log.info(
            "[%s] %s: %d periods, %d/%d fields populated",
            self.source_name,
            self.ticker,
            len(df),
            int(df.notna().any().sum()),
            len(CANONICAL_FIELDS),
        )
        return df

    # --- shared helpers -----------------------------------------------------
    def _coerce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Guarantee exactly CANONICAL_FIELDS as columns, numeric, years desc."""
        # add any missing canonical columns as NaN
        for col in CANONICAL_FIELDS:
            if col not in df.columns:
                df[col] = pd.NA
        # keep only canonical columns, in canonical order
        df = df[CANONICAL_FIELDS].copy()
        # numeric coercion
        for col in CANONICAL_FIELDS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # sort by year descending (most recent first) when index is sortable
        try:
            df = df.sort_index(ascending=False)
        except TypeError:
            pass
        df.index.name = "fiscal_year"
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise LoaderError(
                f"No financial data returned for {self.ticker} from "
                f"'{self.source_name}'. Check the ticker / network access."
            )
        # at least one period must have all 'must be positive' fields valid
        ok = pd.Series(True, index=df.index)
        for field in POSITIVE_REQUIRED:
            ok &= df[field].fillna(0) > 0
        if not ok.any():
            raise LoaderError(
                f"{self.ticker}: no period has valid {POSITIVE_REQUIRED}. "
                "Data is too incomplete to use."
            )
