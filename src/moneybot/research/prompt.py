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

TRIAGE_SYSTEM = "You are a fast triage filter for trading research."


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
    parts = [research_guidance]
    mem_block = _format_memory(memory)
    if mem_block:
        parts.append(mem_block)
    parts.append("")  # blank line before the grounding rule
    parts.append(_GROUNDING_RULE.format(ticker=ticker))
    return "\n".join(parts)


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
