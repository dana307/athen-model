"""
Risk engine (Phase 9).

    from risk import assess
    report = assess(fundamentals)
    print(report.overall_level)      # LOW / MEDIUM / HIGH
    for f in report.findings:
        print(f.label, f.level, f.note)
"""
from __future__ import annotations

from risk.engine import (
    assess,
    RiskAssessment,
    RiskFinding,
    RiskLevel,
    RISK_CHECKS,
)

__all__ = [
    "assess", "RiskAssessment", "RiskFinding", "RiskLevel", "RISK_CHECKS",
]
