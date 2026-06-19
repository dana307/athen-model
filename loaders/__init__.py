"""
Loader registry.

Lets the rest of Athena pick a data source by name without importing concrete
classes:

    from loaders import get_loader
    df = get_loader("yfinance", "RELIANCE").load()
"""
from __future__ import annotations

from loaders.base import BaseLoader, LoaderError
from loaders.yfinance_loader import YFinanceLoader

_REGISTRY: dict[str, type[BaseLoader]] = {
    "yfinance": YFinanceLoader,
}

# screener.in loader is added in a later phase; register it lazily if present.
try:  # pragma: no cover
    from loaders.screener import ScreenerLoader

    _REGISTRY["screener"] = ScreenerLoader
except Exception:
    pass


def get_loader(source: str, ticker: str, **kwargs) -> BaseLoader:
    source = source.lower()
    if source not in _REGISTRY:
        raise LoaderError(
            f"Unknown data source '{source}'. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[source](ticker, **kwargs)


def available_sources() -> list[str]:
    return list(_REGISTRY)


__all__ = ["get_loader", "available_sources", "BaseLoader", "LoaderError"]
