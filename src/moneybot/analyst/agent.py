"""AnalystAgent: rank research signals, then independently confirm the top names.

A thin, strategy-agnostic coordinator (mirrors ResearchAgent). It owns NO entry
logic — the active strategy's `rank` applies the freshness gate + scoring. The
Analyst adds the one hard reasoning step: an independent per-name confirmation
(Opus) before a thesis becomes a TradePlan. All LLM work goes through the
injected LLMClient seam; all price access goes through DataLayer with as_of.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from pydantic import ValidationError

from moneybot.analyst.models import ConfirmationVerdict
from moneybot.analyst.prompt import build_confirm_system, build_confirm_user, confirm_schema
from moneybot.analyst.relative_strength import excess_return
from moneybot.memory.models import MemoryContext

if TYPE_CHECKING:
    from moneybot.config import Settings
    from moneybot.data_layer import DataLayer
    from moneybot.llm.client import LLMClient
    from moneybot.memory.retriever import MemoryRetriever
    from moneybot.strategies.base import Strategy
    from moneybot.strategies.models import CatalystSignal, Proposal


class AnalystAgent:
    def __init__(
        self,
        *,
        data_layer: DataLayer,
        strategy: Strategy,
        llm: LLMClient,
        settings: Settings,
        retriever: MemoryRetriever | None = None,
    ) -> None:
        self.data = data_layer
        self.strategy = strategy
        self.llm = llm
        self.settings = settings
        self.retriever = retriever

    def _relative_strength(self, as_of: date | None = None) -> dict[str, float]:
        """Excess trailing return vs the benchmark for each tradeable name."""
        tf = self.settings.rs_timeframe
        lookback = self.settings.rs_lookback_days
        bench_bars = self.data.get_bars(
            self.data.universe.benchmark, tf, lookback, as_of=as_of
        )
        bench_closes = [] if bench_bars.empty else bench_bars["close"].tolist()
        rs: dict[str, float] = {}
        for symbol in self.data.universe.symbols:
            bars = self.data.get_bars(symbol, tf, lookback, as_of=as_of)
            closes = [] if bars.empty else bars["close"].tolist()
            rs[symbol] = excess_return(closes, bench_closes)
        return rs

    def _memory_context(self, ticker: str) -> MemoryContext:
        if self.retriever is None:
            return MemoryContext()
        return self.retriever.retrieve([ticker], self.data.universe.sector)

    def _confirm(
        self,
        proposal: Proposal,
        signal: CatalystSignal | None,
        memory: MemoryContext,
        *,
        relative_strength: float,
    ) -> ConfirmationVerdict:
        """One Opus call to independently rule on a thesis. Malformed output = reject."""
        try:
            result = self.llm.complete_json(
                model=self.settings.model_analyst,
                system=build_confirm_system(memory, proposal.ticker),
                user=build_confirm_user(
                    proposal, signal, relative_strength=relative_strength
                ),
                schema=confirm_schema(),
            )
            return ConfirmationVerdict.model_validate(result)
        except (ValidationError, ValueError):
            # Never trade on an unparseable/invalid analyst response — treat as rejection.
            return ConfirmationVerdict(
                confirmed=False,
                adjusted_conviction=0.0,
                reasoning="unparseable analyst response",
            )
