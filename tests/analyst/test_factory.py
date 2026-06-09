from moneybot.analyst.agent import AnalystAgent
from moneybot.analyst.factory import build_analyst_agent
from moneybot.config import Settings, TickerMeta, Universe
from moneybot.strategies.catalyst_driven import CatalystDrivenLong


class _FakeLLM:
    def complete_json(self, *, model, system, user, schema):
        return {}


class _FakeData:
    def __init__(self):
        self.universe = Universe(
            sector="semiconductors",
            benchmark="SMH",
            tickers=[TickerMeta(symbol="NVDA")],
        )


def test_build_analyst_agent_resolves_strategy_and_injected_llm():
    agent = build_analyst_agent(
        settings=Settings(strategy="catalyst_driven"),
        data_layer=_FakeData(),
        llm=_FakeLLM(),
    )
    assert isinstance(agent, AnalystAgent)
    assert isinstance(agent.strategy, CatalystDrivenLong)
    # injected fake is used; no AnthropicClient constructed
    assert isinstance(agent.llm, _FakeLLM)
