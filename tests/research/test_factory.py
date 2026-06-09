import pandas as pd

from moneybot.cache import Cache
from moneybot.config import Settings, Universe
from moneybot.data_layer import DataLayer
from moneybot.memory.lessons import LessonStore
from moneybot.memory.retriever import KeyedMemoryRetriever
from moneybot.memory.semantic import SemanticStore
from moneybot.research.agent import ResearchAgent
from moneybot.research.factory import build_research_agent


class _Prices:
    def get_bars(self, ticker, timeframe, lookback, as_of=None):
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])


def test_build_research_agent_wires_active_strategy_and_llm(tmp_path):
    uni = Universe(sector="semiconductors", benchmark="SMH",
                   tickers=[{"symbol": "NVDA"}])
    dl = DataLayer(uni, _Prices(), Cache(tmp_path))
    retriever = KeyedMemoryRetriever(SemanticStore(tmp_path / "s"),
                                     LessonStore(tmp_path / "l"))
    sentinel_llm = object()

    agent = build_research_agent(
        settings=Settings(strategy="catalyst_driven"),
        data_layer=dl, retriever=retriever, llm=sentinel_llm,
    )
    assert isinstance(agent, ResearchAgent)
    assert agent.strategy.name == "catalyst_driven"  # resolved from registry
    assert agent.llm is sentinel_llm  # injected client used, no real SDK built
