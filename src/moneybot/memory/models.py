"""Typed models for the memory subsystem."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Dossier(BaseModel):
    """A versioned piece of semantic memory (sector or per-ticker knowledge)."""

    key: str
    content: str
    version: int
    updated_at: datetime


class JournalEntry(BaseModel):
    """An append-only episodic record (a proposal, fill, outcome, etc.)."""

    entry_id: str
    ts: datetime
    kind: str
    ticker: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class Lesson(BaseModel):
    """A distilled lesson learned from outcomes, applied to a sector or ticker."""

    lesson_id: str
    created_at: datetime
    applies_to: str
    pattern: str
    lesson: str
    confidence: float
    evidence_trades: list[str] = Field(default_factory=list)
    supersedes: str | None = None


class MemoryContext(BaseModel):
    """The relevant memory slice assembled for an agent cycle."""

    dossiers: list[Dossier] = Field(default_factory=list)
    lessons: list[Lesson] = Field(default_factory=list)
