from datetime import date, datetime, timezone

from moneybot.config import Settings
from moneybot.models import Filing, NewsItem
from moneybot.research.agent import ResearchAgent
from moneybot.research.prompt import collect_sources


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
