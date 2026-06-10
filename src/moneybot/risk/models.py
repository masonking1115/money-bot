"""Input and output types for the Risk Engine.

PortfolioState / Position are the account snapshot the orchestrator (Plan 8)
supplies. RiskDecision is the verdict for one TradePlan (approved, downsized, or
vetoed) and always records the rule(s) that fired. RiskAssessment bundles a
cycle's decisions with the halt flag and an optional benchmark hedge.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Position(BaseModel):
    """One open position in the account snapshot."""

    ticker: str
    shares: float  # observed broker quantity; may be fractional, unlike an emitted order
    # Sign convention: long positions are positive. A short would be negative, so
    # `long_market_value` below can sum positives to get gross long exposure.
    market_value: float  # current market value (shares * current price)


class PortfolioState(BaseModel):
    """Account snapshot the Risk Engine sizes against."""

    equity: float = Field(gt=0)  # total account value (cash + positions)
    cash: float
    positions: list[Position] = Field(default_factory=list)
    day_pnl_pct: float = 0.0  # today's P&L as a fraction of starting equity (negative = loss)

    @property
    def long_market_value(self) -> float:
        return sum(p.market_value for p in self.positions if p.market_value > 0)

    @property
    def gross_exposure_pct(self) -> float:
        return self.long_market_value / self.equity if self.equity else 0.0


class RiskDecision(BaseModel):
    """The Risk Engine's verdict on a single TradePlan."""

    ticker: str
    approved: bool
    target_weight: float = 0.0  # approved fraction of equity (0 if vetoed)
    target_dollars: float = 0.0  # shares * reference_price actually deployed
    shares: int = 0  # whole-share order the engine emits (floored from target_dollars/price)
    reference_price: float | None = None
    rules_fired: list[str] = Field(default_factory=list)  # rules that downsized or vetoed
    reasoning: str


class HedgeOrder(BaseModel):
    """An offsetting benchmark position to neutralize sector beta (when enabled)."""

    ticker: str
    side: Literal["short"]
    shares: int
    dollars: float


class RiskAssessment(BaseModel):
    """The full result of assessing one cycle's trade plans."""

    decisions: list[RiskDecision]
    halted: bool = False  # true when a global gate (kill switch / circuit breaker) stopped entries
    hedge: HedgeOrder | None = None

    @property
    def approved(self) -> list[RiskDecision]:
        return [d for d in self.decisions if d.approved]
