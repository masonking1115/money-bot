# Research Agents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build generic, tiered LLM Research agents that read the active strategy's `signal_schema`/`research_guidance`, pull filings/news (point-in-time aware) plus memory context, and emit structured, citation-grounded `CatalystSignal`s.

**Architecture:** A single network seam (`LLMClient` Protocol) isolates all Anthropic calls so no test touches the network — exactly mirroring the providers' `_fetch*` pattern. Prompt assembly and signal validation are **pure functions** (fully unit-tested). The `ResearchAgent` is a thin orchestrator: it retrieves data + memory, runs a cheap **Haiku triage** to pick which documents are worth reading, runs a **Sonnet deep-read** that emits JSON conforming to the strategy's signal schema, then **validates and grounds** every signal (drops empty or hallucinated citations, bounds the ticker, coerces to `CatalystSignal`). The agent is strategy-agnostic: it reads the active strategy's schema/guidance and never hard-codes the catalyst taxonomy. Point-in-time `as_of` flows through to the DataLayer so the agent is backtestable.

**Tech Stack:** Python 3.11+, Pydantic v2, the official `anthropic` SDK (behind a seam), pytest, uv, ruff. No network access in tests.

---

## Design Notes (read before starting)

**Seam discipline (non-negotiable).** The only code that imports `anthropic` or makes a network call is `moneybot/llm/anthropic_client.py`, and even there the raw SDK call lives in one method, `_create_message`, which tests override. Every other module depends on the `LLMClient` Protocol, never on the SDK. This is the same discipline the data providers use (`_fetch*` seams).

**Determinism for backtests.** `signal_id` is a content hash (no clock, no randomness) so the same inputs always produce the same id. `as_of` is threaded into every DataLayer call and the agent never reads data newer than `as_of` (the DataLayer already post-asserts this).

**Citation grounding is a safety feature, not a nicety.** The deep-read prompt gives the model an enumerated SOURCES list, each with a URL. After the model responds, `validate_signals` drops any signal whose evidence cites a URL that was **not** in the provided sources (a hallucinated citation) and any signal with no evidence at all. This bounds the damage from a confident-but-wrong model before a signal can ever reach the Analyst.

**Tiering.** Triage uses `settings.model_triage` (Haiku — cheap filter over headlines/filing metadata). Deep-read uses `settings.model_deep_read` (Sonnet — reads full text, emits signals). The Analyst tier (Opus) is Plan 5, not here.

**Schema wrapping.** `strategy.signal_schema()` describes ONE signal. The agent wraps it for list output: `{"type":"object","properties":{"signals":{"type":"array","items":<signal_schema>}},"required":["signals"]}`. Triage uses its own small schema.

---

## File Structure

**Create:**
- `src/moneybot/llm/__init__.py` — re-exports `LLMClient`, `AnthropicClient`
- `src/moneybot/llm/client.py` — `LLMClient` Protocol (the seam contract)
- `src/moneybot/llm/anthropic_client.py` — `AnthropicClient` adapter (only file importing `anthropic`)
- `src/moneybot/research/__init__.py` — re-exports `ResearchAgent`
- `src/moneybot/research/prompt.py` — pure prompt builders + schema wrappers
- `src/moneybot/research/validate.py` — pure signal validation/grounding + `signal_id`
- `src/moneybot/research/agent.py` — `ResearchAgent` orchestrator (triage → deep-read → validate)
- `tests/llm/__init__.py`
- `tests/llm/test_anthropic_client.py`
- `tests/research/__init__.py`
- `tests/research/test_prompt.py`
- `tests/research/test_validate.py`
- `tests/research/test_agent.py`

**Modify:**
- `pyproject.toml` — add `anthropic` dependency
- `README.md` — Status section: add the Research agents phase

---

## Task 1: LLM client seam + Anthropic adapter

**Files:**
- Modify: `pyproject.toml`
- Create: `src/moneybot/llm/__init__.py`
- Create: `src/moneybot/llm/client.py`
- Create: `src/moneybot/llm/anthropic_client.py`
- Test: `tests/llm/__init__.py`, `tests/llm/test_anthropic_client.py`

- [ ] **Step 1: Add the anthropic dependency**

Edit `pyproject.toml`, adding `anthropic` to `dependencies`:

```toml
dependencies = [
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "pyyaml>=6.0",
    "pandas>=2.2",
    "pyarrow>=15.0",
    "yfinance>=0.2.40",
    "httpx>=0.27",
    "anthropic>=0.40",
]
```

Then run: `uv sync`
Expected: resolves and installs `anthropic`.

- [ ] **Step 2: Write the failing test for the adapter**

Create `tests/llm/__init__.py` (empty file).

Create `tests/llm/test_anthropic_client.py`:

```python
import json

import pytest

from moneybot.llm.anthropic_client import AnthropicClient


class FakeContentBlock:
    def __init__(self, text):
        self.text = text


class FakeMessage:
    def __init__(self, text):
        self.content = [FakeContentBlock(text)]


def test_complete_json_parses_model_text_and_captures_request():
    captured = {}

    class Adapter(AnthropicClient):
        def _create_message(self, **kwargs):
            captured.update(kwargs)
            return FakeMessage(json.dumps({"signals": [{"ticker": "NVDA"}]}))

    client = Adapter(client=object())  # real SDK client never used; seam is overridden
    schema = {"type": "object", "properties": {"signals": {"type": "array"}}}
    out = client.complete_json(
        model="claude-sonnet-4-6",
        system="sys",
        user="usr",
        schema=schema,
    )

    assert out == {"signals": [{"ticker": "NVDA"}]}
    assert captured["model"] == "claude-sonnet-4-6"
    assert captured["system"] == "sys"
    assert captured["messages"] == [{"role": "user", "content": "usr"}]


def test_complete_json_enables_adaptive_thinking_for_non_haiku():
    captured = {}

    class Adapter(AnthropicClient):
        def _create_message(self, **kwargs):
            captured.update(kwargs)
            return FakeMessage('{"ok": true}')

    Adapter(client=object()).complete_json(
        model="claude-sonnet-4-6", system="s", user="u", schema={"type": "object"}
    )
    assert captured.get("thinking") == {"type": "adaptive"}


def test_complete_json_omits_thinking_for_haiku():
    captured = {}

    class Adapter(AnthropicClient):
        def _create_message(self, **kwargs):
            captured.update(kwargs)
            return FakeMessage('{"ok": true}')

    Adapter(client=object()).complete_json(
        model="claude-haiku-4-5", system="s", user="u", schema={"type": "object"}
    )
    assert "thinking" not in captured  # Haiku does not support adaptive thinking


def test_complete_json_raises_on_unparseable_text():
    class Adapter(AnthropicClient):
        def _create_message(self, **kwargs):
            return FakeMessage("not json at all")

    with pytest.raises(ValueError, match="could not parse JSON"):
        Adapter(client=object()).complete_json(
            model="claude-sonnet-4-6", system="s", user="u", schema={"type": "object"}
        )
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/llm/test_anthropic_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'moneybot.llm'`.

- [ ] **Step 4: Write the Protocol**

Create `src/moneybot/llm/client.py`:

```python
"""The LLM client seam: the single contract every agent depends on.

Only moneybot.llm.anthropic_client implements this against the real SDK; all
other code (and every test) depends on this Protocol, so no test hits the
network. Mirrors the providers' _fetch* seam discipline.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    def complete_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Send one structured request and return the parsed JSON object.

        The returned dict conforms to `schema`. Implementations must raise
        ValueError if the model output cannot be parsed as JSON.
        """
        ...
```

- [ ] **Step 5: Write the Anthropic adapter**

Create `src/moneybot/llm/anthropic_client.py`:

```python
"""The only module that talks to the Anthropic API.

The raw SDK call is isolated in `_create_message` so tests override it and no
test hits the network. `complete_json` builds the request (structured-output
config + tiered thinking), calls the seam, and parses the JSON response.

NOTE for the implementer: use the `claude-api` skill to confirm the exact
structured-output request shape for the installed SDK version. The contract
that tests pin is: `_create_message(**kwargs)` returns an object whose
`.content[0].text` is a JSON string; `complete_json` json.loads that text.
"""

from __future__ import annotations

import json
from typing import Any

from moneybot.llm.client import LLMClient


class AnthropicClient(LLMClient):
    def __init__(self, client: Any | None = None, *, max_tokens: int = 4096) -> None:
        if client is None:
            from anthropic import Anthropic  # imported lazily so tests need no key

            client = Anthropic()
        self._client = client
        self._max_tokens = max_tokens

    def _create_message(self, **kwargs: Any) -> Any:
        """Network seam — the only line that calls the SDK. Overridden in tests."""
        return self._client.messages.create(**kwargs)

    def complete_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {
                "format": {"type": "json_schema", "schema": schema, "name": "result"}
            },
        }
        # Haiku supports neither adaptive thinking nor effort; only enable it for
        # the deeper tiers (sonnet/opus).
        if not model.startswith("claude-haiku"):
            kwargs["thinking"] = {"type": "adaptive"}

        message = self._create_message(**kwargs)
        text = message.content[0].text
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"could not parse JSON from model output: {text!r}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
        return parsed
```

Create `src/moneybot/llm/__init__.py`:

```python
"""LLM client seam and the Anthropic adapter."""

from moneybot.llm.anthropic_client import AnthropicClient
from moneybot.llm.client import LLMClient

__all__ = ["LLMClient", "AnthropicClient"]
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/llm/test_anthropic_client.py -v`
Expected: PASS (4 tests).

- [ ] **Step 7: Lint**

Run: `uv run ruff check src/moneybot/llm tests/llm`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src/moneybot/llm tests/llm
git commit -m "feat: add LLM client seam and Anthropic adapter"
```

---

## Task 2: Research prompt assembly (pure)

**Files:**
- Create: `src/moneybot/research/__init__.py`
- Create: `src/moneybot/research/prompt.py`
- Test: `tests/research/__init__.py`, `tests/research/test_prompt.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/research/__init__.py` (empty file).

Create `tests/research/test_prompt.py`:

```python
from datetime import date, datetime, timezone

from moneybot.memory.models import Dossier, Lesson, MemoryContext
from moneybot.models import Filing, NewsItem
from moneybot.research.prompt import (
    SourceDoc,
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/research/test_prompt.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'moneybot.research'`.

- [ ] **Step 3: Write the prompt module**

Create `src/moneybot/research/prompt.py`:

```python
"""Pure prompt assembly for the Research agents.

No LLM calls and no I/O here — just turning typed inputs (strategy guidance,
memory context, source documents) into prompt strings and JSON schemas. This
keeps the prompt content fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from moneybot.memory.models import MemoryContext
from moneybot.models import Filing, NewsItem

_GROUNDING_RULE = (
    "STRICT CITATION RULE: every catalyst's `evidence` MUST quote from and link "
    "to the SOURCES below. You may ONLY cite a `url` that appears in the SOURCES "
    "list — never invent or recall a URL from memory. A catalyst with no evidence "
    "drawn from the SOURCES is invalid and must be omitted. Emit signals ONLY for "
    "ticker {ticker}."
)


@dataclass(frozen=True)
class SourceDoc:
    """One enumerated source document handed to the model."""

    index: int
    kind: Literal["filing", "news"]
    url: str
    headline: str   # short label for triage (form type / title)
    date: str       # ISO date string
    body: str       # full text for deep read ("" if unavailable)


def collect_sources(filings: list[Filing], news: list[NewsItem]) -> list[SourceDoc]:
    """Flatten filings + news into a single indexed, citable source list."""
    sources: list[SourceDoc] = []
    for f in filings:
        sources.append(
            SourceDoc(
                index=len(sources), kind="filing", url=f.url,
                headline=f.form_type, date=f.filed_at.isoformat(),
                body=f.raw_text or "",
            )
        )
    for n in news:
        sources.append(
            SourceDoc(
                index=len(sources), kind="news", url=n.url,
                headline=n.title, date=n.published_at.date().isoformat(),
                body=n.summary or "",
            )
        )
    return sources


def wrap_signals_schema(single_signal_schema: dict[str, Any]) -> dict[str, Any]:
    """Wrap a strategy's single-signal schema into a list-output schema."""
    return {
        "type": "object",
        "properties": {
            "signals": {"type": "array", "items": single_signal_schema}
        },
        "required": ["signals"],
    }


def wrap_triage_schema() -> dict[str, Any]:
    """Schema for the cheap triage pass: which source indices warrant a deep read."""
    return {
        "type": "object",
        "properties": {
            "relevant_indices": {"type": "array", "items": {"type": "integer"}}
        },
        "required": ["relevant_indices"],
    }


def build_triage_user(ticker: str, sources: list[SourceDoc]) -> str:
    """Cheap headline-only listing for the triage tier (no bodies)."""
    lines = [
        f"Ticker: {ticker}",
        "Below are recent documents. Return the indices worth a full read for "
        "fresh, material, bullish catalysts.",
        "",
    ]
    for s in sources:
        lines.append(f"[{s.index}] ({s.kind}, {s.date}) {s.headline}")
    return "\n".join(lines)


def _format_memory(memory: MemoryContext) -> str:
    if not memory.dossiers and not memory.lessons:
        return ""
    parts = ["", "OPERATOR KNOWLEDGE (use to judge materiality):"]
    for d in memory.dossiers:
        parts.append(f"- [{d.key}] {d.content}")
    for lsn in memory.lessons:
        parts.append(f"- LESSON ({lsn.applies_to}, conf {lsn.confidence}): {lsn.lesson}")
    return "\n".join(parts)


def build_deep_read_system(
    research_guidance: str, memory: MemoryContext, ticker: str
) -> str:
    """System prompt: strategy guidance + operator memory + grounding rule."""
    return "\n".join(
        [
            research_guidance,
            _format_memory(memory),
            "",
            _GROUNDING_RULE.format(ticker=ticker),
        ]
    )


def build_deep_read_user(
    ticker: str, sources: list[SourceDoc], *, max_body_chars: int = 4000
) -> str:
    """User prompt for the deep read: enumerated SOURCES with bodies + URLs."""
    lines = [f"Ticker: {ticker}", "", "SOURCES:"]
    for s in sources:
        body = s.body[:max_body_chars]
        lines.append(
            f"[{s.index}] kind={s.kind} date={s.date} url={s.url}\n"
            f"    headline: {s.headline}\n"
            f"    body: {body}"
        )
    return "\n".join(lines)
```

Create `src/moneybot/research/__init__.py`:

```python
"""Research agents: read filings/news and emit citation-grounded signals."""
```

(The `ResearchAgent` re-export is added in Task 5.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/research/test_prompt.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/moneybot/research tests/research`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/moneybot/research tests/research
git commit -m "feat: add pure research prompt assembly"
```

---

## Task 3: Signal validation + grounding (pure)

**Files:**
- Create: `src/moneybot/research/validate.py`
- Test: `tests/research/test_validate.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/research/test_validate.py`:

```python
from moneybot.research.validate import make_signal_id, validate_signals
from moneybot.strategies.models import CatalystSignal


def _raw(ticker="NVDA", url="https://sec/1", evidence=None, **over):
    base = {
        "ticker": ticker,
        "category": "guidance",
        "direction": "bullish",
        "materiality": 0.8,
        "freshness_days": 2,
        "conviction": 0.7,
        "evidence": evidence if evidence is not None
        else [{"source": "8-K", "quote": "Raised guidance", "url": url}],
        "thesis": "Guidance raised.",
    }
    base.update(over)
    return base


ALLOWED = {"https://sec/1", "https://news/1"}


def test_valid_signal_passes_and_is_coerced_to_model():
    out = validate_signals([_raw()], ticker="NVDA", allowed_urls=ALLOWED)
    assert len(out) == 1
    assert isinstance(out[0], CatalystSignal)
    assert out[0].signal_id is not None  # id assigned


def test_signal_with_no_evidence_is_dropped():
    out = validate_signals([_raw(evidence=[])], ticker="NVDA", allowed_urls=ALLOWED)
    assert out == []


def test_signal_citing_unknown_url_is_dropped():
    bad = _raw(evidence=[{"source": "x", "quote": "q", "url": "https://hallucinated/9"}])
    out = validate_signals([bad], ticker="NVDA", allowed_urls=ALLOWED)
    assert out == []


def test_signal_for_wrong_ticker_is_dropped():
    out = validate_signals([_raw(ticker="AMD")], ticker="NVDA", allowed_urls=ALLOWED)
    assert out == []


def test_signal_with_any_ungrounded_citation_is_dropped():
    # one good, one hallucinated url in the same signal -> drop whole signal
    mixed = _raw(evidence=[
        {"source": "8-K", "quote": "q", "url": "https://sec/1"},
        {"source": "x", "quote": "q", "url": "https://hallucinated/9"},
    ])
    out = validate_signals([mixed], ticker="NVDA", allowed_urls=ALLOWED)
    assert out == []


def test_malformed_signal_is_skipped_not_raised():
    # materiality out of range -> pydantic rejects; validator skips it
    out = validate_signals([_raw(materiality=5.0)], ticker="NVDA", allowed_urls=ALLOWED)
    assert out == []


def test_signal_id_is_deterministic_and_content_addressed():
    a = validate_signals([_raw()], ticker="NVDA", allowed_urls=ALLOWED)[0]
    b = validate_signals([_raw()], ticker="NVDA", allowed_urls=ALLOWED)[0]
    assert a.signal_id == b.signal_id  # same inputs -> same id
    # different thesis -> different id
    c = validate_signals([_raw(thesis="Different.")], ticker="NVDA", allowed_urls=ALLOWED)[0]
    assert c.signal_id != a.signal_id


def test_make_signal_id_changes_with_evidence_urls():
    s1 = CatalystSignal(ticker="NVDA", category="guidance", direction="bullish",
                        materiality=0.8, freshness_days=2, conviction=0.7,
                        evidence=[{"source": "s", "quote": "q", "url": "https://sec/1"}],
                        thesis="t")
    s2 = s1.model_copy(update={"evidence": [
        {"source": "s", "quote": "q", "url": "https://news/1"}]})
    assert make_signal_id(s1) != make_signal_id(s2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/research/test_validate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'moneybot.research.validate'`.

- [ ] **Step 3: Write the validation module**

Create `src/moneybot/research/validate.py`:

```python
"""Pure validation + grounding for raw signals emitted by the deep-read tier.

Three guards, in order: (1) coerce the raw dict into a CatalystSignal (Pydantic
rejects bad ranges/enums); (2) bound the ticker to the one we asked about;
(3) ground every citation — drop the whole signal if it has no evidence or cites
any URL we did not provide (anti-hallucination). Surviving signals get a
deterministic, content-addressed signal_id.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

from pydantic import ValidationError

from moneybot.strategies.models import CatalystSignal


def make_signal_id(signal: CatalystSignal) -> str:
    """Content hash over the fields that define a distinct catalyst."""
    urls = "|".join(sorted(e.url for e in signal.evidence))
    payload = f"{signal.ticker}|{signal.category}|{signal.thesis}|{urls}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def validate_signals(
    raw_signals: Iterable[dict[str, Any]],
    *,
    ticker: str,
    allowed_urls: set[str],
) -> list[CatalystSignal]:
    """Return only the well-formed, on-ticker, fully-grounded signals."""
    valid: list[CatalystSignal] = []
    for raw in raw_signals:
        try:
            signal = CatalystSignal.model_validate(raw)
        except ValidationError:
            continue  # malformed -> skip, never raise on model output

        if signal.ticker != ticker:
            continue
        if not signal.evidence:
            continue
        if any(e.url not in allowed_urls for e in signal.evidence):
            continue  # any ungrounded citation invalidates the signal

        valid.append(signal.model_copy(update={"signal_id": make_signal_id(signal)}))
    return valid
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/research/test_validate.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/moneybot/research/validate.py tests/research/test_validate.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/moneybot/research/validate.py tests/research/test_validate.py
git commit -m "feat: add signal validation and citation grounding"
```

---

## Task 4: Triage step (Haiku tier)

**Files:**
- Create: `src/moneybot/research/agent.py` (first slice — the triage method only)
- Test: `tests/research/test_agent.py` (triage tests)

This task introduces the `ResearchAgent` shell and its triage method. Task 5 adds the deep-read orchestration to the same file.

- [ ] **Step 1: Write the failing triage tests**

Create `tests/research/test_agent.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/research/test_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'moneybot.research.agent'`.

- [ ] **Step 3: Write the agent shell + triage**

Create `src/moneybot/research/agent.py`:

```python
"""ResearchAgent: orchestrate triage -> deep-read -> validate for a strategy.

The agent is a thin, strategy-agnostic coordinator. It owns NO catalyst logic —
it reads the active strategy's signal_schema/research_guidance and delegates all
LLM work to the injected LLMClient seam (so tests never hit the network).
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

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
```

(`date` and `Any` imports are used by the deep-read method added in Task 5; if ruff flags them as unused after this task, add the deep-read method from Task 5 in the same change so they are used. To keep Task 4 lint-clean on its own, omit the `date`/`Any` imports now and add them in Task 5.)

To keep this task lint-clean standalone, the imports block should be exactly:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from moneybot.research.prompt import (
    SourceDoc,
    build_triage_user,
    wrap_triage_schema,
)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/research/test_agent.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/moneybot/research/agent.py tests/research/test_agent.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/moneybot/research/agent.py tests/research/test_agent.py
git commit -m "feat: add ResearchAgent triage step (Haiku tier)"
```

---

## Task 5: ResearchAgent orchestration (deep-read + universe)

**Files:**
- Modify: `src/moneybot/research/agent.py` (add deep-read + research methods)
- Modify: `src/moneybot/research/__init__.py` (re-export `ResearchAgent`)
- Test: `tests/research/test_agent.py` (add orchestration tests)

- [ ] **Step 1: Write the failing orchestration tests**

Append to `tests/research/test_agent.py`:

```python
import pandas as pd

from moneybot.cache import Cache
from moneybot.config import Universe
from moneybot.data_layer import DataLayer
from moneybot.memory.lessons import LessonStore
from moneybot.memory.retriever import KeyedMemoryRetriever
from moneybot.memory.semantic import SemanticStore
from moneybot.strategies.catalyst_driven import CatalystDrivenLong


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/research/test_agent.py -v`
Expected: FAIL — `AttributeError: 'ResearchAgent' object has no attribute 'research_ticker'`.

- [ ] **Step 3: Add the deep-read + orchestration methods**

Replace the imports block at the top of `src/moneybot/research/agent.py` with:

```python
from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from moneybot.research.prompt import (
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
```

Then add these methods to the `ResearchAgent` class (after `_triage`):

```python
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
```

Note: `build_triage_user` and `wrap_triage_schema` remain used by `_triage`; `SourceDoc` is used in type hints. All imports are now exercised.

Update `src/moneybot/research/__init__.py`:

```python
"""Research agents: read filings/news and emit citation-grounded signals."""

from moneybot.research.agent import ResearchAgent

__all__ = ["ResearchAgent"]
```

- [ ] **Step 4: Run the full research + agent test suite**

Run: `uv run pytest tests/research/ -v`
Expected: PASS (all prompt, validate, and agent tests).

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/moneybot/research tests/research`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/moneybot/research/agent.py src/moneybot/research/__init__.py tests/research/test_agent.py
git commit -m "feat: add ResearchAgent deep-read and universe orchestration"
```

---

## Task 6: Wiring, README, final review + merge

**Files:**
- Create: `src/moneybot/research/factory.py`
- Test: `tests/research/test_factory.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing factory test**

The factory builds a wired `ResearchAgent` from `Settings` + the active strategy registry, injecting the real `AnthropicClient` (or an override for tests). It must NOT construct the real SDK client during the test.

Create `tests/research/test_factory.py`:

```python
from datetime import date, datetime, timezone

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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/research/test_factory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'moneybot.research.factory'`.

- [ ] **Step 3: Write the factory**

Create `src/moneybot/research/factory.py`:

```python
"""Wire a ResearchAgent from settings: resolve the active strategy and the LLM.

The real AnthropicClient is constructed lazily and only when no `llm` override
is supplied, so tests inject a fake and never touch the SDK or require an API key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from moneybot.research.agent import ResearchAgent
from moneybot.strategies import registry

if TYPE_CHECKING:
    from moneybot.config import Settings
    from moneybot.data_layer import DataLayer
    from moneybot.llm.client import LLMClient
    from moneybot.memory.retriever import MemoryRetriever


def build_research_agent(
    *,
    settings: Settings,
    data_layer: DataLayer,
    retriever: MemoryRetriever,
    llm: LLMClient | None = None,
) -> ResearchAgent:
    if llm is None:
        from moneybot.llm.anthropic_client import AnthropicClient

        llm = AnthropicClient()
    strategy = registry.get(settings.strategy)
    return ResearchAgent(
        data_layer=data_layer,
        retriever=retriever,
        strategy=strategy,
        llm=llm,
        settings=settings,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/research/test_factory.py -v`
Expected: PASS.

Note: the factory references `moneybot.strategies.registry.get`; the active strategy is auto-registered by importing `moneybot.strategies`. Confirm `moneybot/strategies/__init__.py` registers `CatalystDrivenLong` on import (it does, per Plan 4). If `registry.get` raises `unknown strategy`, add `import moneybot.strategies  # noqa: F401` at the top of `factory.py` to trigger registration.

- [ ] **Step 5: Update the README Status section**

Add a Research agents bullet to the Status section in `README.md` (match the existing phrasing/format of the Phase 1–4 bullets):

```markdown
- **Research agents** — generic tiered LLM agents (Haiku triage → Sonnet deep-read)
  that read the active strategy's signal schema + research guidance, pull filings/news
  (point-in-time aware) plus operator memory, and emit citation-grounded `CatalystSignal`s.
  All Anthropic calls sit behind an `LLMClient` seam, so no test touches the network.
```

- [ ] **Step 6: Run the full suite + lint**

Run: `uv run pytest -q`
Expected: PASS (all prior 102 tests + the new research/llm tests).

Run: `uv run ruff check src tests`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/moneybot/research/factory.py tests/research/test_factory.py README.md
git commit -m "feat: add research agent factory wiring and update README"
```

- [ ] **Step 8: Final whole-implementation review**

Dispatch a final code-reviewer subagent over the entire Research agents implementation (all of Tasks 1–6). Specifically verify:
- **No network in tests:** nothing under `tests/` imports `anthropic` or constructs a real `Anthropic()` client; the seam (`_create_message`) is always overridden or the `llm` is injected.
- **Seam isolation:** only `anthropic_client.py` imports `anthropic`.
- **Grounding holds:** a signal citing any URL outside the provided sources is dropped; an empty-evidence signal is dropped; an off-ticker signal is dropped.
- **Point-in-time:** `as_of` reaches every DataLayer call in `research_ticker`; the agent reads no data newer than `as_of`.
- **Determinism:** `signal_id` is content-addressed (no clock/randomness).
- **Strategy-agnostic:** the agent reads `signal_schema()`/`research_guidance()` and hard-codes no catalyst taxonomy.

Fix any issues the reviewer raises, re-run the suite, and commit.

- [ ] **Step 9: Merge to main**

Use the superpowers:finishing-a-development-branch skill. Verify tests pass, then merge the plan branch to `main` with `--no-ff` and push.

---

## Self-Review (against the strategy spec + architecture)

- **Spec Section 3 (catalyst signal):** the agent emits `CatalystSignal` with mandatory evidence; ungrounded/empty citations are dropped → covered by Task 3 + Task 5.
- **Spec Section 2.5 (generic executors):** the agent reads the active strategy's `signal_schema`/`research_guidance` and is otherwise taxonomy-free → covered by Task 5 (deep-read) + Task 6 (factory resolves active strategy).
- **Tiering (Haiku triage → Sonnet deep-read):** Task 4 + Task 5 use `settings.model_triage` / `settings.model_deep_read`.
- **Memory integration (Plan 3):** `_memory_context` pulls dossiers/lessons via the retriever and the system prompt embeds them → Task 5 + Task 2.
- **Point-in-time discipline:** `as_of` threaded end-to-end → Task 5 tests pin it.
- **No-network-in-tests invariant:** the `LLMClient` seam + injected fakes → Tasks 1, 4, 5, 6.
- **Determinism for backtests:** content-addressed `signal_id` → Task 3.
- **Type consistency:** `complete_json(model, system, user, schema) -> dict`, `SourceDoc(index, kind, url, headline, date, body)`, `validate_signals(raw, *, ticker, allowed_urls)`, `ResearchAgent(*, data_layer, retriever, strategy, llm, settings)` are used identically across all tasks.
