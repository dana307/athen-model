"""Phase 10 tests — report builder + DOCX/PDF rendering (offline)."""
from __future__ import annotations

import numpy as np
import pytest

from loaders.yfinance_loader import YFinanceLoader
from reports import build_report, generate_report
from tests import fixtures


def _load(bundle):
    class L(YFinanceLoader):
        def fetch_raw(self):
            return bundle
    return L("X").load()


@pytest.fixture
def healthy():
    return _load(fixtures.raw_bundle())


def _peers():
    return {
        "PEERBIG": (_load(fixtures.scaled_bundle(2.0, 0.1)), 3000.0),
        "PEERSML": (_load(fixtures.scaled_bundle(0.5)), 2500.0),
    }


def test_report_object(healthy):
    r = build_report(healthy, ticker="RELIANCE", market_price=2900.0,
                     beta=1.1, peers=_peers())
    # scenarios present and ordered bull > base > bear in value
    assert set(r.scenarios) == {"Bull", "Base", "Bear"}
    assert (r.scenarios["Bull"]["price"] > r.scenarios["Base"]["price"]
            > r.scenarios["Bear"]["price"])
    assert r.recommendation in {"BUY", "HOLD", "REDUCE"}
    assert r.thesis and r.key_risks
    # fair value blends DCF + comps
    assert not np.isnan(r.fair_value)


def test_recommendation_logic(healthy):
    r = build_report(healthy, ticker="X", market_price=2900.0)
    # intrinsic well below market here -> REDUCE
    assert r.upside < 0
    assert r.recommendation == "REDUCE"


def test_generate_docx_pdf(healthy, tmp_path):
    written = generate_report(
        healthy, ticker="RELIANCE", market_price=2900.0, beta=1.1,
        peers=_peers(), out_dir=tmp_path, formats=("docx", "pdf"),
    )
    for fmt in ("docx", "pdf"):
        p = tmp_path / f"RELIANCE_Athena.{fmt}"
        assert p.exists() and p.stat().st_size > 5000   # non-trivial file


def test_charts_generated(healthy, tmp_path):
    from reports import charts
    r = build_report(healthy, ticker="RELIANCE", market_price=2900.0)
    paths = charts.generate_all(r, tmp_path)
    for key in ("revenue", "montecarlo", "scenarios", "sensitivity"):
        assert key in paths
        assert (tmp_path / f"RELIANCE_{_suffix(key)}").exists() or \
               __import__("os").path.exists(paths[key])


def _suffix(key):
    return {"revenue": "revenue.png", "montecarlo": "mc.png",
            "scenarios": "scenarios.png", "sensitivity": "sensitivity.png"}[key]
