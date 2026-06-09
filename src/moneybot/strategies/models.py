"""Types shared across the strategy framework.

Proposal / ExitPlan / StrategyParams are generic to all strategies. Evidence /
CatalystSignal are the signal types for the catalyst-driven strategy (future
strategies may define their own signal types).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """A citation backing a catalyst claim."""

    source: str
    quote: str
    url: str


class CatalystSignal(BaseModel):
    """A structured catalyst the Research agents emit for a ticker."""

    ticker: str
    category: Literal["guidance", "demand", "supply", "policy"]
    direction: Literal["bullish", "bearish", "neutral"]
    materiality: float = Field(ge=0.0, le=1.0)
    freshness_days: int = Field(ge=0)
    conviction: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence]
    thesis: str
    signal_id: str | None = None


class Proposal(BaseModel):
    """A ranked entry candidate produced by a strategy. Sizing is the Risk
    Engine's job — the strategy only proposes the name, conviction, and score."""

    ticker: str
    action: Literal["buy"]
    conviction: float = Field(ge=0.0, le=1.0)
    thesis: str
    score: float
    signal_ref: str | None = None


class ExitPlan(BaseModel):
    """Mechanical exit configuration plus thesis-check guidance for a strategy."""

    max_hold_days: int
    stop_loss_pct: float
    profit_target_pct: float
    thesis_check_guidance: str


class StrategyParams(BaseModel):
    """Tunable parameters for a strategy (defaults are backtest starting points)."""

    freshness_window_days: int = 5
    max_hold_days: int = 10
    stop_loss_pct: float = 0.08
    profit_target_pct: float = 0.20
    max_position_pct: float = 0.10
    max_sector_exposure_pct: float = 0.60
    hedge_enabled: bool = False
