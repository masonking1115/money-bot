"""Wire the whole bot into an Orchestrator from settings.

The caller supplies the data layer and memory retriever (their construction —
providers, cache, universe, memory stores — is outside this plan). Everything
else is built here from the existing component factories, sharing one injected
clock so the journal's timestamps line up with the cycle (and a later backtest
can replay dated cycles). The LLM is optional: omit it in production to lazily
construct the real client, inject a fake in tests.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

from moneybot.analyst.factory import build_analyst_agent
from moneybot.execution.factory import build_execution_adapter
from moneybot.memory.journal import JournalStore
from moneybot.orchestrator.engine import Orchestrator
from moneybot.orchestrator.market_hours import is_market_open
from moneybot.orchestrator.portfolio import SodEquityStore
from moneybot.research.factory import build_research_agent
from moneybot.risk.factory import build_risk_engine
from moneybot.strategies import registry

if TYPE_CHECKING:
    from moneybot.config import Settings
    from moneybot.data_layer import DataLayer
    from moneybot.llm.client import LLMClient
    from moneybot.memory.retriever import MemoryRetriever


def build_orchestrator(
    *,
    settings: Settings,
    data_layer: DataLayer,
    retriever: MemoryRetriever,
    llm: LLMClient | None = None,
    clock: Callable[[], datetime] | None = None,
    market_open: Callable[[datetime], bool] = is_market_open,
) -> Orchestrator:
    clock = clock or (lambda: datetime.now(timezone.utc))

    research = build_research_agent(
        settings=settings, data_layer=data_layer, retriever=retriever, llm=llm
    )
    analyst = build_analyst_agent(
        settings=settings, data_layer=data_layer, retriever=retriever, llm=llm
    )
    risk = build_risk_engine(settings=settings, data_layer=data_layer)
    execution = build_execution_adapter(settings=settings)
    journal = JournalStore(settings.data_dir, clock=clock)
    sod_equity = SodEquityStore(settings.data_dir)
    strategy = registry.get(settings.strategy)

    return Orchestrator(
        settings=settings,
        data_layer=data_layer,
        research=research,
        analyst=analyst,
        risk=risk,
        execution=execution,
        journal=journal,
        sod_equity=sod_equity,
        strategy=strategy,
        clock=clock,
        market_open=market_open,
    )
