from datetime import date, datetime, timezone

from moneybot.memory.models import Dossier, Lesson, MemoryContext
from moneybot.models import Filing, NewsItem
from moneybot.research.prompt import (
    SourceDoc,  # noqa: F401  (imported to assert it is a public export)
    build_deep_read_system,
    build_deep_read_user,
    build_triage_user,
    collect_sources,
    wrap_signals_schema,
    wrap_triage_schema,
)


def _filing():
    return Filing(
        ticker="NVDA", form_type="8-K", filed_at=date(2026, 6, 5),
        accession_no="a-1", url="https://sec/1", raw_text="Raised FY guidance materially.",
    )


def _news():
    return NewsItem(
        ticker="NVDA", title="Big design win", url="https://news/1",
        published_at=datetime(2026, 6, 6, tzinfo=timezone.utc), source="wire",
        summary="Hyperscaler picks NVDA.",
    )


def test_collect_sources_indexes_filings_and_news():
    sources = collect_sources([_filing()], [_news()])
    assert [s.url for s in sources] == ["https://sec/1", "https://news/1"]
    assert sources[0].index == 0 and sources[1].index == 1
    assert sources[0].kind == "filing" and sources[1].kind == "news"


def test_wrap_signals_schema_wraps_single_signal_schema_in_array():
    single = {"type": "object", "properties": {"ticker": {"type": "string"}}}
    wrapped = wrap_signals_schema(single)
    assert wrapped["properties"]["signals"]["items"] == single
    assert wrapped["required"] == ["signals"]


def test_wrap_triage_schema_expects_integer_indices():
    schema = wrap_triage_schema()
    assert schema["properties"]["relevant_indices"]["items"]["type"] == "integer"


def test_triage_user_lists_each_source_with_its_index():
    sources = collect_sources([_filing()], [_news()])
    text = build_triage_user("NVDA", sources)
    assert "[0]" in text and "[1]" in text
    assert "8-K" in text and "Big design win" in text
    # triage must NOT include raw bodies (it is a cheap headline filter)
    assert "Raised FY guidance materially." not in text


def test_deep_read_system_includes_guidance_memory_and_grounding_rule():
    mem = MemoryContext(
        dossiers=[Dossier(key="sector:semis", content="NVDA drives AI capex.",
                          version=1, updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc))],
        lessons=[Lesson(lesson_id="l1", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                        applies_to="ticker:NVDA", pattern="beats priced in",
                        lesson="NVDA beats are often priced in.", confidence=0.7)],
    )
    sys = build_deep_read_system("GUIDANCE_TEXT", mem, "NVDA")
    assert "GUIDANCE_TEXT" in sys
    assert "NVDA drives AI capex." in sys
    assert "NVDA beats are often priced in." in sys
    assert "NVDA" in sys
    # the anti-hallucination rule must be present
    assert "only cite" in sys.lower()


def test_deep_read_user_includes_indexed_sources_with_bodies_and_urls():
    sources = collect_sources([_filing()], [_news()])
    text = build_deep_read_user("NVDA", sources)
    assert "https://sec/1" in text and "https://news/1" in text
    assert "Raised FY guidance materially." in text  # deep read DOES include bodies
    assert "[0]" in text and "[1]" in text


def test_deep_read_user_truncates_long_bodies():
    big = Filing(ticker="NVDA", form_type="10-Q", filed_at=date(2026, 6, 5),
                 accession_no="a-2", url="https://sec/2", raw_text="x" * 10_000)
    sources = collect_sources([big], [])
    text = build_deep_read_user("NVDA", sources, max_body_chars=500)
    assert "x" * 500 in text
    assert "x" * 501 not in text
