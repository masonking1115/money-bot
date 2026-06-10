"""Orchestrator: run one full trading cycle end-to-end.

Order of operations each cycle: kill-switch gate -> market-hours gate -> mechanical
exits (close triggered longs) -> research -> analyst -> portfolio snapshot -> risk
engine -> entry execution -> reconcile, journaling each step. Every collaborator is
injected, so tests use fakes and nothing hits the network or an LLM. The clock is
injected (no fabricated time); the cycle_id is derived from it so a re-run within
the same hour is idempotent at the broker/store.

This task implements construction + the two global gates; exits and the entry
pipeline are added next.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from moneybot.orchestrator.models import CycleResult
from moneybot.risk.kill_switch import kill_switch_active

if TYPE_CHECKING:
    from datetime import datetime

    from moneybot.config import Settings
    from moneybot.data_layer import DataLayer
    from moneybot.orchestrator.portfolio import SodEquityStore
    from moneybot.strategies.base import Strategy


class Orchestrator:
    def __init__(
        self,
        *,
        settings: Settings,
        data_layer: DataLayer,
        research,
        analyst,
        risk,
        execution,
        journal,
        sod_equity: SodEquityStore,
        strategy: Strategy,
        clock: Callable[[], datetime],
        market_open: Callable[[datetime], bool],
    ) -> None:
        self.settings = settings
        self.data = data_layer
        self.research = research
        self.analyst = analyst
        self.risk = risk
        self.execution = execution
        self.journal = journal
        self.sod_equity = sod_equity
        self.strategy = strategy
        self._clock = clock
        self._market_open = market_open

    def run_cycle(self, as_of=None) -> CycleResult:
        now = self._clock()
        cycle_id = now.strftime("%Y-%m-%dT%H")

        if kill_switch_active(self.settings):
            self.journal.append("halt", None, {"reason": "kill_switch"})
            return CycleResult(status="halted", reason="kill_switch", cycle_id=cycle_id)

        if not self._market_open(now):
            self.journal.append("skip", None, {"reason": "market_closed"})
            return CycleResult(status="skipped", reason="market_closed", cycle_id=cycle_id)

        # Exits + entry pipeline added in Tasks 8-9.
        return CycleResult(status="completed", cycle_id=cycle_id)
