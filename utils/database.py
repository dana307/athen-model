"""
SQLite persistence for Athena.

Design choices
--------------
- One row per (ticker, source, fiscal_year, field) in a long/tidy table.
  This is forgiving as the schema evolves: adding a canonical field needs no
  migration, and querying any subset is trivial.
- `save_fundamentals` is an upsert keyed on (ticker, source, fiscal_year,
  field), so re-running a ticker refreshes rather than duplicates.
- `load_fundamentals` pivots back into the canonical wide DataFrame
  (index = fiscal_year desc, columns = CANONICAL_FIELDS).

Later phases can move to PostgreSQL by swapping the connection; the SQL here is
deliberately standard.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import settings
from utils.schema import CANONICAL_FIELDS
from utils.logging import get_logger

log = get_logger("database")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS fundamentals (
    ticker        TEXT    NOT NULL,
    source        TEXT    NOT NULL,
    fiscal_year   INTEGER NOT NULL,
    field         TEXT    NOT NULL,
    value         REAL,
    updated_at    TEXT    NOT NULL,
    PRIMARY KEY (ticker, source, fiscal_year, field)
);
CREATE INDEX IF NOT EXISTS idx_fund_ticker ON fundamentals (ticker);
"""


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else settings.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    return conn


def save_fundamentals(
    df: pd.DataFrame,
    ticker: str,
    source: str,
    db_path: Path | str | None = None,
) -> int:
    """Upsert a canonical fundamentals DataFrame. Returns rows written."""
    ticker = ticker.strip().upper()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for year, row in df.iterrows():
        for field in CANONICAL_FIELDS:
            val = row.get(field)
            if pd.isna(val):
                continue
            rows.append((ticker, source, int(year), field, float(val), now))

    with connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO fundamentals
                (ticker, source, fiscal_year, field, value, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, source, fiscal_year, field)
            DO UPDATE SET value = excluded.value,
                          updated_at = excluded.updated_at
            """,
            rows,
        )
        conn.commit()
    log.info("saved %d datapoints for %s (%s)", len(rows), ticker, source)
    return len(rows)


def load_fundamentals(
    ticker: str,
    source: str | None = None,
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    """Read fundamentals back as the canonical wide DataFrame."""
    ticker = ticker.strip().upper()
    query = "SELECT fiscal_year, field, value FROM fundamentals WHERE ticker = ?"
    params: list = [ticker]
    if source:
        query += " AND source = ?"
        params.append(source)

    with connect(db_path) as conn:
        long = pd.read_sql_query(query, conn, params=params)

    if long.empty:
        return pd.DataFrame(columns=CANONICAL_FIELDS)

    wide = long.pivot_table(
        index="fiscal_year", columns="field", values="value", aggfunc="last"
    )
    for col in CANONICAL_FIELDS:
        if col not in wide.columns:
            wide[col] = pd.NA
    wide = wide[CANONICAL_FIELDS].sort_index(ascending=False)
    wide.index.name = "fiscal_year"
    return wide
