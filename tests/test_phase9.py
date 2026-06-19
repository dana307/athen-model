"""Phase 9 tests — risk engine (offline)."""
from __future__ import annotations

import pytest

from loaders.yfinance_loader import YFinanceLoader
from risk import assess, RiskLevel, RISK_CHECKS
from tests import fixtures


def _load(bundle):
    class L(YFinanceLoader):
        def fetch_raw(self):
            return bundle
    return L("X").load()


@pytest.fixture
def healthy():
    return _load(fixtures.raw_bundle())


@pytest.fixture
def stressed():
    return _load(fixtures.stressed_bundle())


def _by_key(report):
    return {f.key: f for f in report.findings}


# --- overall ratings --------------------------------------------------------
def test_healthy_company_is_low_risk(healthy):
    report = assess(healthy)
    assert report.overall_level == RiskLevel.LOW
    assert report.not_assessed == []          # all six checks ran


def test_stressed_company_is_high_risk(stressed):
    report = assess(stressed)
    assert report.overall_level == RiskLevel.HIGH
    assert report.risk_score > 2.0


def test_all_six_checks_run(healthy):
    report = assess(healthy)
    assert len(report.findings) == len(RISK_CHECKS) == 6


# --- individual checks on the stressed company -----------------------------
def test_leverage_high(stressed):
    f = _by_key(assess(stressed))["leverage"]
    assert f.level == RiskLevel.HIGH
    assert f.value > 3.0                       # net debt / EBITDA


def test_margins_deteriorating(stressed):
    f = _by_key(assess(stressed))["margins"]
    assert f.level == RiskLevel.HIGH
    assert f.value < 0                          # margin fell


def test_negative_fcf(stressed):
    f = _by_key(assess(stressed))["fcf"]
    assert f.level == RiskLevel.HIGH
    assert f.value < 0


def test_rising_receivables(stressed):
    f = _by_key(assess(stressed))["receivables"]
    assert f.level == RiskLevel.HIGH


def test_debt_spike(stressed):
    f = _by_key(assess(stressed))["debt_spike"]
    assert f.level == RiskLevel.HIGH
    assert f.value == pytest.approx(0.5, abs=0.01)   # +50% YoY


# --- graceful degradation ---------------------------------------------------
def test_missing_data_marks_not_assessed(healthy):
    bundle = fixtures.raw_bundle()
    bundle["balance"] = bundle["balance"].drop(index="Accounts Receivable")
    df = _load(bundle)
    report = assess(df)
    assert "receivables" in report.not_assessed
    # other checks still ran
    assert any(f.key == "leverage" for f in report.findings)
