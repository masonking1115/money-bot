"""Result types for one orchestrator cycle.

ExitSignal is a triggered mechanical exit (stop/target/time-stop) the orchestrator
turns into a sell order. CycleResult is the structured summary one run_cycle call
returns — rich enough for a later observability layer to render, without this layer
owning any presentation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from moneybot.execution.models import Fill, ReconciliationResult


class ExitSignal(BaseModel):
    """A mechanical exit fired on an open long position."""

    ticker: str
    shares: int = Field(gt=0)  # whole shares to sell (the open quantity)
    reason: Literal["stop_loss", "profit_target", "time_stop"]
    reference_price: float


class CycleResult(BaseModel):
    """Structured outcome of one orchestrator cycle."""

    status: Literal["completed", "halted", "skipped"]
    reason: str = ""  # why halted/skipped (e.g. "kill_switch", "market_closed")
    cycle_id: str = ""
    plans_proposed: int = 0  # TradePlans the analyst produced
    entry_fills: list[Fill] = Field(default_factory=list)
    exit_fills: list[Fill] = Field(default_factory=list)
    halted_by_risk: bool = False  # a global risk gate (kill switch / circuit breaker) fired
    reconciliation: ReconciliationResult | None = None
