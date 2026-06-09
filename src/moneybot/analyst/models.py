"""Analyst output types — generic across strategies.

ConfirmationVerdict is the structured result of the Analyst's independent
confirmation LLM call. TradePlan is what the Analyst emits to the Risk Engine:
a ranked, confirmed entry recommendation carrying its exit plan. The Analyst
proposes target names and conviction only — it never sizes or places trades.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from moneybot.strategies.models import ExitPlan


class ConfirmationVerdict(BaseModel):
    """The Analyst's independent ruling on a single ranked thesis."""

    confirmed: bool
    adjusted_conviction: float = Field(ge=0.0, le=1.0)
    reasoning: str
    risk_flags: list[str] = Field(default_factory=list)


class TradePlan(BaseModel):
    """A confirmed entry recommendation. Sizing/approval is the Risk Engine's job."""

    ticker: str
    action: Literal["buy"]
    conviction: float = Field(ge=0.0, le=1.0)
    thesis: str
    score: float
    signal_ref: str | None = None
    exit_plan: ExitPlan
    analyst_note: str
    risk_flags: list[str] = Field(default_factory=list)
