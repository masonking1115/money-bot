"""Pure prompt assembly for the Analyst's independent-confirmation call.

No LLM calls and no I/O — typed inputs (a ranked Proposal, its backing
CatalystSignal, the relative-strength reading, operator memory) become prompt
strings and a JSON schema. Keeps the prompt content fully unit-testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from moneybot.memory.models import MemoryContext

if TYPE_CHECKING:
    from moneybot.strategies.models import CatalystSignal, Proposal

_ANALYST_SYSTEM = """\
You are a skeptical sector analyst reviewing a single ranked entry candidate for {ticker}.
A research pass already proposed it; your job is to INDEPENDENTLY decide whether the
catalyst genuinely justifies opening a new long right now, or whether it is stale, weak,
already priced in, or contradicted. Do not rubber-stamp the proposal.

Return:
- confirmed: true only if you would open a new long on this thesis today.
- adjusted_conviction (0-1): your own conviction, which may differ from the proposal's.
- reasoning: a concise justification grounded in the catalyst and the evidence shown.
- risk_flags: any concerns (e.g. earnings imminent, thin evidence, crowded trade)."""


def confirm_schema() -> dict[str, Any]:
    """JSON schema for the ConfirmationVerdict the Analyst must return."""
    return {
        "type": "object",
        "properties": {
            "confirmed": {"type": "boolean"},
            "adjusted_conviction": {"type": "number", "minimum": 0, "maximum": 1},
            "reasoning": {"type": "string"},
            "risk_flags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["confirmed", "adjusted_conviction", "reasoning"],
    }


def _format_memory(memory: MemoryContext) -> str:
    if not memory.dossiers and not memory.lessons:
        return ""
    parts = ["", "PRIOR KNOWLEDGE & LESSONS (weigh these in your ruling):"]
    for d in memory.dossiers:
        parts.append(f"- [{d.key}] {d.content}")
    for lsn in memory.lessons:
        parts.append(
            f"- LESSON ({lsn.applies_to}, conf {lsn.confidence}): {lsn.lesson}"
        )
    return "\n".join(parts)


def build_confirm_system(memory: MemoryContext, ticker: str) -> str:
    """System prompt: the skeptical-analyst role + operator memory."""
    parts = [_ANALYST_SYSTEM.format(ticker=ticker)]
    mem_block = _format_memory(memory)
    if mem_block:
        parts.append(mem_block)
    return "\n".join(parts)


def build_confirm_user(
    proposal: Proposal,
    signal: CatalystSignal | None,
    *,
    relative_strength: float,
) -> str:
    """User prompt: the candidate's thesis, catalyst, evidence, score, and RS reading."""
    lines = [
        f"Ticker: {proposal.ticker}",
        f"Proposed thesis: {proposal.thesis}",
        f"Rank score: {proposal.score}",
        f"Relative strength vs benchmark (excess trailing return): {relative_strength}",
        "",
    ]
    if signal is not None:
        lines.append(
            f"Catalyst: category={signal.category} direction={signal.direction} "
            f"materiality={signal.materiality} freshness_days={signal.freshness_days} "
            f"research_conviction={signal.conviction}"
        )
        lines.append("Evidence:")
        for e in signal.evidence:
            lines.append(f"  - ({e.source}) \"{e.quote}\" {e.url}")
    else:
        lines.append("Catalyst: (backing signal unavailable — judge on the thesis alone)")
    lines.append("")
    lines.append(
        "Decide whether to confirm this long. Return your verdict in the required schema."
    )
    return "\n".join(lines)
