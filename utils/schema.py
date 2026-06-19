"""
Canonical financial schema for Athena.

Every loader, regardless of source (yfinance, screener.in, XBRL, ...), must
return data conforming to this schema. That decoupling is what lets the
valuation engine stay source-agnostic: it only ever sees CANONICAL_FIELDS.

Convention
----------
A "fundamentals" table is a pandas DataFrame where:
    - the index is the fiscal-year label (e.g. 2024, 2023, ...), most recent first
    - the columns are exactly CANONICAL_FIELDS
    - all monetary values are in absolute INR (not crores/millions)
"""
from __future__ import annotations

# Canonical field names — the single source of truth for the rest of Athena.
# Order matters only for display.
CANONICAL_FIELDS: list[str] = [
    "revenue",            # Total operating revenue / net sales
    "ebitda",             # Earnings before interest, tax, depreciation, amortization
    "ebit",               # Operating profit (EBITDA - D&A)
    "pat",                # Profit after tax (net income to shareholders)
    "depreciation",       # Depreciation & amortization
    "capex",              # Capital expenditure (reported as a positive outflow)
    "working_capital",    # Net working capital = current assets - current liabilities
    "receivables",        # Accounts receivable (trade debtors)
    "inventory",          # Inventory / stock
    "cash",               # Cash & cash equivalents
    "debt",               # Total debt (short + long term borrowings)
    "total_equity",       # Shareholders' equity (book value)
    "shares_outstanding", # Diluted shares outstanding
]

# Human-readable labels for reports/dashboards.
FIELD_LABELS: dict[str, str] = {
    "revenue": "Revenue",
    "ebitda": "EBITDA",
    "ebit": "EBIT",
    "pat": "Profit After Tax",
    "depreciation": "Depreciation & Amortization",
    "capex": "Capital Expenditure",
    "working_capital": "Net Working Capital",
    "receivables": "Accounts Receivable",
    "inventory": "Inventory",
    "cash": "Cash & Equivalents",
    "debt": "Total Debt",
    "total_equity": "Shareholders' Equity",
    "shares_outstanding": "Shares Outstanding",
}

# Fields that must be strictly positive for a row to be considered valid.
POSITIVE_REQUIRED: list[str] = ["revenue", "shares_outstanding"]
