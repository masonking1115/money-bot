"""Typed inputs and outputs for the backtest harness."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class BacktestConfig(BaseModel):
    """Operator-supplied backtest parameters."""

    start: date
    end: date
    timeframe: str = "1d"
    starting_cash: float = Field(default=100_000.0, gt=0)
    mode: Literal["record", "replay"] = "record"
    use_agents: bool = True  # False -> mechanical-only (no AI; exits/seeded positions only)

    @model_validator(mode="after")
    def _end_after_start(self) -> BacktestConfig:
        if self.end < self.start:
            raise ValueError("end must be on or after start")
        return self


class EquityPoint(BaseModel):
    """Mark-to-market account state at the close of one simulated day."""

    day: date
    equity: float
    cash: float
    n_positions: int


class TradeRecord(BaseModel):
    """A closed round-trip (FIFO-matched buy -> sell), long-only for phase 1."""

    ticker: str
    qty: int
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    exit_reason: str = ""


class PerformanceMetrics(BaseModel):
    """Headline numbers the go-live gate is judged against."""

    total_return: float
    cagr: float
    max_drawdown: float  # positive magnitude of worst peak-to-trough decline
    sharpe: float
    win_rate: float
    n_trades: int
    final_equity: float
    benchmark_return: float
    benchmark_final_equity: float


class BacktestReport(BaseModel):
    config: BacktestConfig
    equity_curve: list[EquityPoint] = Field(default_factory=list)
    trades: list[TradeRecord] = Field(default_factory=list)
    metrics: PerformanceMetrics
    notes: list[str] = Field(default_factory=list)
