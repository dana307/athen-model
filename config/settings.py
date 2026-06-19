"""
Central configuration for Athena.

Anything that a user might reasonably want to tune (paths, default data
source, valuation defaults) lives here so the rest of the codebase never
hard-codes magic numbers.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "output"
DB_PATH = DATA_DIR / "athena.sqlite"

for _d in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
# Which loader to use by default. "yfinance" is the free, reliable source we
# start with; "screener" will be added later as a swappable implementation.
DEFAULT_SOURCE = "yfinance"

# Indian tickers on Yahoo Finance use the ".NS" (NSE) suffix.
DEFAULT_EXCHANGE_SUFFIX = ".NS"

# ---------------------------------------------------------------------------
# Valuation defaults (used in later phases; defined here for one source of truth)
# ---------------------------------------------------------------------------
RISK_FREE_RATE = 0.070          # ~10Y Indian G-Sec, override per run
EQUITY_RISK_PREMIUM = 0.060     # India ERP, override per run
DEFAULT_TAX_RATE = 0.25         # Indian corporate tax (new regime, approx)
TERMINAL_GROWTH = 0.045         # long-run nominal growth assumption
FORECAST_YEARS = 5

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
# All monetary values are normalized to this unit internally.
REPORTING_CURRENCY = "INR"
