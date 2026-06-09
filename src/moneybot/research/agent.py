"""ResearchAgent: orchestrate triage -> deep-read -> validate for a strategy.

The agent is a thin, strategy-agnostic coordinator. It owns NO catalyst logic —
it reads the active strategy's signal_schema/research_guidance and delegates all
LLM work to the injected LLMClient seam (so tests never hit the network).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from moneybot.research.prompt import (
    SourceDoc,
    build_triage_user,
    wrap_triage_schema,
)

if TYPE_CHECKING:
    from moneybot.config import Settings
    from moneybot.data_layer import DataLayer
    from moneybot.llm.client import LLMClient
    from moneybot.memory.retriever import MemoryRetriever
    from moneybot.strategies.base import Strategy


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
            system="You are a fast triage filter for trading research.",
            user=build_triage_user(ticker, sources),
            schema=wrap_triage_schema(),
        )
        wanted = {int(i) for i in result.get("relevant_indices", [])}
        return [s for s in sources if s.index in wanted]
