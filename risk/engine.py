"""
Phase 9 — Risk Engine.

Reads the derived-metrics history and flags the classic financial-distress
signals, each as a Low / Medium / High finding with a plain-English note:

    excess leverage · falling margins · negative free cash flow ·
    rising receivables · inventory buildup · debt spikes

Each check is a small registered function (same pluggable pattern as the
loaders and forecasters), so adding a new red flag is one decorated function.
The engine runs them all and rolls the findings up into an overall rating
(weakest-link: the overall level is the worst individual finding).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import pandas as pd

from utils.metrics import add_derived_metrics
from utils.logging import get_logger

log = get_logger("risk")


class RiskLevel(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

    def __str__(self) -> str:
        return self.name


@dataclass
class RiskFinding:
    key: str
    label: str
    level: RiskLevel
    note: str
    value: float | None = None


# registry: key -> check function
RISK_CHECKS: dict = {}


def risk_check(key: str, label: str):
    def deco(fn):
        fn._key, fn._label = key, label
        RISK_CHECKS[key] = fn
        return fn
    return deco


def _trend(series: pd.Series):
    """(earliest, latest) of a metric, ascending by year; NaN-safe."""
    s = series.dropna()
    if len(s) < 2:
        return None, None
    return float(s.iloc[0]), float(s.iloc[-1])


# --------------------------------------------------------------------------
# Checks. Each receives the ascending derived-metrics frame and returns a
# RiskFinding, or None when the data is insufficient to assess.
# --------------------------------------------------------------------------
@risk_check("leverage", "Excess leverage")
def _leverage(m: pd.DataFrame) -> RiskFinding | None:
    nd = m["net_debt"].dropna()
    eb = m["ebitda"].dropna()
    if nd.empty or eb.empty:
        return None
    net_debt, ebitda = float(nd.iloc[-1]), float(eb.iloc[-1])
    if ebitda <= 0:
        return RiskFinding("leverage", "Excess leverage", RiskLevel.HIGH,
                           "EBITDA is non-positive — leverage cannot be serviced from operations.",
                           value=np.nan)
    ratio = net_debt / ebitda
    if ratio < 1.5:
        lvl, desc = RiskLevel.LOW, "comfortable"
    elif ratio < 3.0:
        lvl, desc = RiskLevel.MEDIUM, "elevated"
    else:
        lvl, desc = RiskLevel.HIGH, "stretched"
    return RiskFinding("leverage", "Excess leverage", lvl,
                       f"Net debt / EBITDA = {ratio:.1f}x ({desc}).", value=ratio)


@risk_check("margins", "Falling margins")
def _margins(m: pd.DataFrame) -> RiskFinding | None:
    first, last = _trend(m["ebitda_margin"])
    if first is None:
        return None
    delta = last - first           # change in EBITDA margin (pp, as fraction)
    if delta >= -0.01:
        lvl, desc = RiskLevel.LOW, "stable or expanding"
    elif delta >= -0.03:
        lvl, desc = RiskLevel.MEDIUM, "softening"
    else:
        lvl, desc = RiskLevel.HIGH, "deteriorating"
    return RiskFinding("margins", "Falling margins", lvl,
                       f"EBITDA margin moved {delta*100:+.1f}pp over the period "
                       f"({first*100:.1f}% → {last*100:.1f}%, {desc}).",
                       value=delta)


@risk_check("fcf", "Negative free cash flow")
def _fcf(m: pd.DataFrame) -> RiskFinding | None:
    fcff = m["fcff"].dropna()
    if fcff.empty:
        return None
    recent = fcff.iloc[-min(3, len(fcff)):]
    n_neg = int((recent < 0).sum())
    latest = float(fcff.iloc[-1])
    if latest < 0:
        lvl = RiskLevel.HIGH
        note = f"Latest FCFF is negative (₹{latest/1e7:,.0f} cr); {n_neg} of last {len(recent)} yrs negative."
    elif n_neg > 0:
        lvl = RiskLevel.MEDIUM
        note = f"FCFF positive now but negative in {n_neg} of last {len(recent)} yrs."
    else:
        lvl = RiskLevel.LOW
        note = f"FCFF positive across the last {len(recent)} years."
    return RiskFinding("fcf", "Negative free cash flow", lvl, note, value=latest)


@risk_check("receivables", "Rising receivables")
def _receivables(m: pd.DataFrame) -> RiskFinding | None:
    first, last = _trend(m["receivables_pct_sales"])
    if first is None:
        return None
    delta = last - first
    if delta <= 0.02:
        lvl, desc = RiskLevel.LOW, "in line with sales"
    elif delta <= 0.05:
        lvl, desc = RiskLevel.MEDIUM, "outpacing sales"
    else:
        lvl, desc = RiskLevel.HIGH, "sharply outpacing sales (possible collection/quality issues)"
    return RiskFinding("receivables", "Rising receivables", lvl,
                       f"Receivables/sales moved {delta*100:+.1f}pp "
                       f"({first*100:.1f}% → {last*100:.1f}%, {desc}).", value=delta)


@risk_check("inventory", "Inventory buildup")
def _inventory(m: pd.DataFrame) -> RiskFinding | None:
    first, last = _trend(m["inventory_pct_sales"])
    if first is None:
        return None
    delta = last - first
    if delta <= 0.02:
        lvl, desc = RiskLevel.LOW, "controlled"
    elif delta <= 0.05:
        lvl, desc = RiskLevel.MEDIUM, "building"
    else:
        lvl, desc = RiskLevel.HIGH, "building rapidly (possible demand softness/obsolescence)"
    return RiskFinding("inventory", "Inventory buildup", lvl,
                       f"Inventory/sales moved {delta*100:+.1f}pp "
                       f"({first*100:.1f}% → {last*100:.1f}%, {desc}).", value=delta)


@risk_check("debt_spike", "Debt spike")
def _debt_spike(m: pd.DataFrame) -> RiskFinding | None:
    debt = m["debt"].dropna()
    if len(debt) < 2 or debt.iloc[-2] <= 0:
        return None
    yoy = debt.iloc[-1] / debt.iloc[-2] - 1
    if yoy < 0.20:
        lvl, desc = RiskLevel.LOW, "modest"
    elif yoy < 0.40:
        lvl, desc = RiskLevel.MEDIUM, "notable"
    else:
        lvl, desc = RiskLevel.HIGH, "sharp"
    return RiskFinding("debt_spike", "Debt spike", lvl,
                       f"Total debt changed {yoy*100:+.0f}% YoY ({desc}).", value=yoy)


# --------------------------------------------------------------------------
@dataclass
class RiskAssessment:
    findings: list[RiskFinding]
    overall_level: RiskLevel
    risk_score: float                       # mean of assessed finding levels
    not_assessed: list[str] = field(default_factory=list)

    def summary(self) -> str:
        counts = {lvl: sum(f.level == lvl for f in self.findings) for lvl in RiskLevel}
        return (f"Overall {self.overall_level} "
                f"(H{counts[RiskLevel.HIGH]}/M{counts[RiskLevel.MEDIUM]}/"
                f"L{counts[RiskLevel.LOW]}, score {self.risk_score:.2f})")


def assess(fundamentals: pd.DataFrame) -> RiskAssessment:
    """Run every registered check and roll up an overall rating."""
    m = add_derived_metrics(fundamentals).sort_index(ascending=True)
    findings, not_assessed = [], []
    for key, fn in RISK_CHECKS.items():
        finding = fn(m)
        if finding is None:
            not_assessed.append(key)
        else:
            findings.append(finding)

    if findings:
        overall = RiskLevel(max(f.level for f in findings))   # weakest link
        score = float(np.mean([int(f.level) for f in findings]))
    else:
        overall, score = RiskLevel.LOW, float("nan")

    res = RiskAssessment(findings=findings, overall_level=overall,
                         risk_score=score, not_assessed=not_assessed)
    log.info("Risk | %s", res.summary())
    return res
