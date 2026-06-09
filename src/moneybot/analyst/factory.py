"""Wire an AnalystAgent from settings: resolve the active strategy and the LLM.

The real AnthropicClient is constructed lazily and only when no `llm` override
is supplied, so tests inject a fake and never touch the SDK or require an API key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import moneybot.strategies  # noqa: F401  -- import for side-effect: registers strategies
from moneybot.analyst.agent import AnalystAgent
from moneybot.strategies import registry

if TYPE_CHECKING:
    from moneybot.config import Settings
    from moneybot.data_layer import DataLayer
    from moneybot.llm.client import LLMClient
    from moneybot.memory.retriever import MemoryRetriever


def build_analyst_agent(
    *,
    settings: Settings,
    data_layer: DataLayer,
    retriever: MemoryRetriever | None = None,
    llm: LLMClient | None = None,
) -> AnalystAgent:
    if llm is None:
        from moneybot.llm.anthropic_client import AnthropicClient

        llm = AnthropicClient()
    strategy = registry.get(settings.strategy)
    return AnalystAgent(
        data_layer=data_layer,
        strategy=strategy,
        llm=llm,
        settings=settings,
        retriever=retriever,
    )
