"""ResearchAgent: orchestrate triage -> deep-read -> validate for a strategy.

The agent is a thin, strategy-agnostic coordinator. It owns NO catalyst logic —
it reads the active strategy's signal_schema/research_guidance and delegates all
LLM work to the injected LLMClient seam (so tests never hit the network).
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from moneybot.research.prompt import (
    TRIAGE_SYSTEM,
    SourceDoc,
    build_deep_read_system,
    build_deep_read_user,
    build_triage_user,
    collect_sources,
    wrap_signals_schema,
    wrap_triage_schema,
)
from moneybot.research.validate import validate_signals

if TYPE_CHECKING:
    from moneybot.config import Settings
    from moneybot.data_layer import DataLayer
    from moneybot.llm.client import LLMClient
    from moneybot.memory.models import MemoryContext
    from moneybot.memory.retriever import MemoryRetriever
    from moneybot.strategies.base import Strategy
    from moneybot.strategies.models import CatalystSignal


class ResearchAgent:
    def __init__(
        self,
        *,
        data_layer: DataLayer | None,
        retriever: MemoryRetriever | None,
        strategy: Strategy | None,
        llm: LLMClient,
        settings: Settings,
    ) -> None:
        self.data = data_layer
        self.retriever = retriever
        self.strategy = strategy
        self.llm = llm
        self.settings = settings

    def _triage(self, ticker: str, sources: list[SourceDoc]) -> list[SourceDoc]:
        """Cheap Haiku pass: pick which sources warrant a full read."""
        if not sources:
            return []
        result = self.llm.complete_json(
            model=self.settings.model_triage,
            system=TRIAGE_SYSTEM,
            user=build_triage_user(ticker, sources),
            schema=wrap_triage_schema(),
        )
        wanted = {int(i) for i in result.get("relevant_indices", [])}
        return [s for s in sources if s.index in wanted]

    def _memory_context(self, ticker: str) -> MemoryContext:
        from moneybot.memory.models import MemoryContext

        if self.retriever is None:
            return MemoryContext()
        return self.retriever.retrieve([ticker], self.data.universe.sector)

    def _deep_read(
        self, ticker: str, sources: list[SourceDoc], memory: MemoryContext
    ) -> list[CatalystSignal]:
        """Sonnet pass: read full sources, emit citation-grounded signals."""
        if not sources:
            return []
        schema = wrap_signals_schema(self.strategy.signal_schema())
        result = self.llm.complete_json(
            model=self.settings.model_deep_read,
            system=build_deep_read_system(
                self.strategy.research_guidance(), memory, ticker
            ),
            user=build_deep_read_user(ticker, sources),
            schema=schema,
        )
        allowed_urls = {s.url for s in sources}
        return validate_signals(
            result.get("signals", []), ticker=ticker, allowed_urls=allowed_urls
        )

    def research_ticker(
        self, ticker: str, as_of: date | None = None
    ) -> list[CatalystSignal]:
        """Full pipeline for one name: gather -> triage -> deep-read -> validate."""
        filings = self.data.get_filings(ticker, as_of=as_of)
        news = self.data.get_news(ticker, as_of=as_of)
        sources = collect_sources(filings, news)
        selected = self._triage(ticker, sources)
        if not selected:
            return []
        memory = self._memory_context(ticker)
        return self._deep_read(ticker, selected, memory)

    def research_universe(
        self, as_of: date | None = None
    ) -> dict[str, list[CatalystSignal]]:
        """Research every name in the universe (the benchmark ETF is skipped)."""
        return {
            symbol: self.research_ticker(symbol, as_of=as_of)
            for symbol in self.data.universe.symbols
        }
