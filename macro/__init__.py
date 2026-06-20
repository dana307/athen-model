"""
Macro package (Phase 14).

    from macro import stress_test, available_scenarios
    res = stress_test(fundamentals, market_price=2900, sector="Energy")
    res.table        # intrinsic value & % impact per scenario
"""
from __future__ import annotations

from macro.scenarios import (
    SCENARIOS, StressScenario, MacroShock, available_scenarios,
)
from macro.stress import stress_test, StressResult

__all__ = [
    "stress_test", "StressResult",
    "SCENARIOS", "StressScenario", "MacroShock", "available_scenarios",
]
