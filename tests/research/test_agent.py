from datetime import date, datetime, timezone

import pandas as pd

from moneybot.cache import Cache
from moneybot.config import Settings, Universe
from moneybot.data_layer import DataLayer
from moneybot.memory.lessons import LessonStore
from moneybot.memory.retriever import KeyedMemoryRetriever
from moneybot.memory.semantic import SemanticStore
from moneybot.models import Filing, NewsItem
from moneybot.research.agent import ResearchAgent
from moneybot.research.prompt import build_triage_user, collect_sources
from moneybot.strategies.catalyst_driven import CatalystDrivenLong


class ScriptedLLM:
    """A fake LLMClient: returns queued responses and records every request."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    def complete_json(self, *, model, system, user, schema):
        self.requests.append(
            {"model": model, "system": system, "user": user, "schema": schema}
        )
        return self._responses.pop(0)


def _settings():
    return Settings(model_triage="claude-haiku-4-5", model_deep_read="claude-sonnet-4-6")


def _sources():
    f = Filing(ticker="NVDA", form_type="8-K", filed_at=date(2026, 6, 5),
               accession_no="a-1", url="https://sec/1", raw_text="body")
    n = NewsItem(ticker="NVDA", title="news", url="https://news/1",
                 published_at=datetime(2026, 6, 6, tzinfo=timezone.utc), source="w")
    return collect_sources([f], [n])


def test_triage_uses_triage_model_and_returns_selected_sources():
    llm = ScriptedLLM([{"relevant_indices": [1]}])
    agent = ResearchAgent(data_layer=None, retriever=None,
                          strategy=None, llm=llm, settings=_settings())
    selected = agent._triage("NVDA", _sources())
    assert [s.index for s in selected] == [1]
    assert llm.requests[0]["model"] == "claude-haiku-4-5"  # cheap tier
    assert llm.requests[0]["user"] == build_triage_user("NVDA", _sources())


def test_triage_ignores_out_of_range_indices():
    llm = ScriptedLLM([{"relevant_indices": [1, 99]}])  # 99 is not a real source
    agent = ResearchAgent(data_layer=None, retriever=None,
                          strategy=None, llm=llm, settings=_settings())
    selected = agent._triage("NVDA", _sources())
    assert [s.index for s in selected] == [1]


def test_triage_with_no_sources_skips_llm_call():
    llm = ScriptedLLM([])  # no responses queued; must not be called
    agent = ResearchAgent(data_layer=None, retriever=None,
                          strategy=None, llm=llm, settings=_settings())
    assert agent._triage("NVDA", []) == []
    assert llm.requests == []


class _Prices:
    def get_bars(self, ticker, timeframe, lookback, as_of=None):
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])


class _Filings:
    def get_recent_filings(self, ticker, types=None, since=None, as_of=None):
        return [Filing(ticker=ticker, form_type="8-K", filed_at=date(2026, 6, 5),
                       accession_no="a-1", url="https://sec/1",
                       raw_text="Raised FY guidance materially.")]


class _News:
    def get_news(self, query, since=None, as_of=None):
        return [NewsItem(ticker=query, title="Design win", url="https://news/1",
                         published_at=datetime(2026, 6, 6, tzinfo=timezone.utc),
                         source="wire", summary="Hyperscaler picks it.")]


def _datalayer(tmp_path):
    uni = Universe(sector="semiconductors", benchmark="SMH",
                   tickers=[{"symbol": "NVDA"}, {"symbol": "AMD"}])
    return DataLayer(uni, _Prices(), Cache(tmp_path),
                     filings_provider=_Filings(), news_provider=_News())


def _retriever(tmp_path):
    return KeyedMemoryRetriever(SemanticStore(tmp_path / "sem"),
                                LessonStore(tmp_path / "les"))


def _good_signal(url="https://sec/1"):
    return {
        "ticker": "NVDA", "category": "guidance", "direction": "bullish",
        "materiality": 0.8, "freshness_days": 2, "conviction": 0.7,
        "evidence": [{"source": "8-K", "quote": "Raised guidance", "url": url}],
        "thesis": "FY guidance raised.",
    }


def test_research_ticker_returns_grounded_signals(tmp_path):
    # response 0 = triage (read both), response 1 = deep-read signals
    llm = ScriptedLLM([
        {"relevant_indices": [0, 1]},
        {"signals": [_good_signal()]},
    ])
    agent = ResearchAgent(
        data_layer=_datalayer(tmp_path), retriever=_retriever(tmp_path),
        strategy=CatalystDrivenLong(), llm=llm, settings=_settings(),
    )
    signals = agent.research_ticker("NVDA")
    assert len(signals) == 1
    assert signals[0].ticker == "NVDA"
    assert signals[0].signal_id is not None
    # deep-read used the deep-read tier and the strategy's wrapped schema
    deep_req = llm.requests[1]
    assert deep_req["model"] == "claude-sonnet-4-6"
    assert deep_req["schema"]["required"] == ["signals"]


def test_research_ticker_drops_hallucinated_citation(tmp_path):
    llm = ScriptedLLM([
        {"relevant_indices": [0]},
        {"signals": [_good_signal(url="https://hallucinated/9")]},
    ])
    agent = ResearchAgent(
        data_layer=_datalayer(tmp_path), retriever=_retriever(tmp_path),
        strategy=CatalystDrivenLong(), llm=llm, settings=_settings(),
    )
    assert agent.research_ticker("NVDA") == []  # ungrounded -> dropped


def test_research_ticker_with_no_relevant_sources_skips_deep_read(tmp_path):
    llm = ScriptedLLM([{"relevant_indices": []}])  # triage selects nothing
    agent = ResearchAgent(
        data_layer=_datalayer(tmp_path), retriever=_retriever(tmp_path),
        strategy=CatalystDrivenLong(), llm=llm, settings=_settings(),
    )
    assert agent.research_ticker("NVDA") == []
    assert len(llm.requests) == 1  # no deep-read call made


def test_research_universe_covers_all_symbols(tmp_path):
    # 2 symbols x (triage + deep-read) = 4 responses
    llm = ScriptedLLM([
        {"relevant_indices": [0]}, {"signals": [_good_signal()]},
        {"relevant_indices": [0]}, {"signals": []},
    ])
    agent = ResearchAgent(
        data_layer=_datalayer(tmp_path), retriever=_retriever(tmp_path),
        strategy=CatalystDrivenLong(), llm=llm, settings=_settings(),
    )
    out = agent.research_universe()
    assert set(out.keys()) == {"NVDA", "AMD"}
    assert len(out["NVDA"]) == 1
    assert out["AMD"] == []


def test_research_ticker_passes_as_of_to_datalayer(tmp_path):
    captured = {}

    class _AsOfFilings:
        def get_recent_filings(self, ticker, types=None, since=None, as_of=None):
            captured["filings_as_of"] = as_of
            return []

    class _AsOfNews:
        def get_news(self, query, since=None, as_of=None):
            captured["news_as_of"] = as_of
            return []

    uni = Universe(sector="semiconductors", benchmark="SMH",
                   tickers=[{"symbol": "NVDA"}])
    dl = DataLayer(uni, _Prices(), Cache(tmp_path),
                   filings_provider=_AsOfFilings(), news_provider=_AsOfNews())
    llm = ScriptedLLM([])  # no sources -> triage returns [], no LLM call
    agent = ResearchAgent(data_layer=dl, retriever=_retriever(tmp_path),
                          strategy=CatalystDrivenLong(), llm=llm, settings=_settings())
    agent.research_ticker("NVDA", as_of=date(2026, 6, 7))
    assert captured["filings_as_of"] == date(2026, 6, 7)
    assert captured["news_as_of"] == date(2026, 6, 7)
