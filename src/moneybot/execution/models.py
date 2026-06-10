"""Typed inputs and outputs for the Execution Adapter.

OrderRequest is what the adapter asks a broker to place; Fill is what comes
back. PositionRecord is the shared shape for both a broker-reported holding and
the bot's own stored belief (qty is signed: long positive, short negative).
ReconciliationResult/Discrepancy report drift between the two — report-only.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Side = Literal["buy", "sell", "short", "cover"]


class OrderRequest(BaseModel):
    """One order the adapter asks a broker to place (phase-1: market orders)."""

    client_order_id: str  # deterministic idempotency key, e.g. "<cycle>:<ticker>:<side>"
    ticker: str
    side: Side
    quantity: int = Field(gt=0)  # whole shares
    order_type: Literal["market"] = "market"
    reference_price: float | None = None  # paper broker fills here; live broker ignores it


class Fill(BaseModel):
    """The broker's response to a placed order."""

    client_order_id: str
    broker_order_id: str
    ticker: str
    side: Side
    status: Literal["filled", "accepted", "rejected"]
    filled_qty: int = 0
    avg_price: float = 0.0
    ts: datetime
    reason: str = ""  # populated on rejection / partial


class PositionRecord(BaseModel):
    """A single holding. qty is signed: long positive, short negative."""

    ticker: str
    qty: float
    avg_price: float


class AccountSnapshot(BaseModel):
    """Top-line account figures the broker reports."""

    equity: float
    cash: float


class Discrepancy(BaseModel):
    """One position where the bot's stored belief differs from the broker."""

    ticker: str
    stored_qty: float
    broker_qty: float


class ReconciliationResult(BaseModel):
    """Outcome of comparing stored positions against broker positions."""

    in_sync: bool
    discrepancies: list[Discrepancy] = Field(default_factory=list)
