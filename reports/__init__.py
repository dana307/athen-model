"""
Reports package (Phase 10).

    from reports import build_report, generate_report
    paths = generate_report(fundamentals, ticker="RELIANCE", market_price=2900,
                            formats=("docx", "pdf"))
"""
from __future__ import annotations

from pathlib import Path

from config import settings
from reports.builder import build_report, ResearchReport
from reports import charts as _charts


def generate_report(
    fundamentals,
    *,
    ticker: str,
    market_price: float,
    formats=("docx", "pdf"),
    out_dir: Path | None = None,
    **kwargs,
) -> dict:
    """Build the analysis, render charts, and write the requested formats.

    Returns {format: path}.
    """
    out_dir = Path(out_dir) if out_dir else settings.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(fundamentals, ticker=ticker,
                          market_price=market_price, **kwargs)
    report.charts = _charts.generate_all(report, out_dir / "charts")

    written = {}
    if "docx" in formats:
        from reports.docx_report import build_docx
        written["docx"] = str(build_docx(report, out_dir / f"{ticker.upper()}_Athena.docx"))
    if "pdf" in formats:
        from reports.pdf_report import build_pdf
        written["pdf"] = str(build_pdf(report, out_dir / f"{ticker.upper()}_Athena.pdf"))
    return written


__all__ = ["build_report", "ResearchReport", "generate_report"]
