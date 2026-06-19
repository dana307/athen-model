"""
screener.in loader — STUB (planned for a later phase).

The hybrid strategy: build the whole pipeline on yfinance first, then add this
richer Indian source as a drop-in. It will subclass BaseLoader exactly like
YFinanceLoader, so nothing downstream changes when we switch sources.

To implement later:
    - fetch_raw(): GET https://www.screener.in/company/<TICKER>/consolidated/
      (requests + a logged-in session / API), parse the financial tables.
    - normalize(): map screener's rows onto utils.schema.CANONICAL_FIELDS.
"""
from __future__ import annotations

from loaders.base import BaseLoader, LoaderError


class ScreenerLoader(BaseLoader):  # pragma: no cover - not yet implemented
    source_name = "screener"

    def fetch_raw(self):
        raise LoaderError(
            "ScreenerLoader is not implemented yet. Use source='yfinance'. "
            "Planned for a later phase of the Athena roadmap."
        )

    def normalize(self, raw):
        raise NotImplementedError
