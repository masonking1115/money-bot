"""Typed domain models shared across the data layer and agents."""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, computed_field


class Bar(BaseModel):
    """A single OHLCV price bar."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = Field(ge=0)


class Filing(BaseModel):
    """An SEC filing (or other regulatory document)."""

    ticker: str
    form_type: str
    filed_at: date
    accession_no: str
    url: str
    raw_text: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_hash(self) -> str:
        payload = f"{self.accession_no}|{self.url}|{self.raw_text or ''}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class NewsItem(BaseModel):
    """A news headline/article reference."""

    ticker: str | None = None
    title: str
    url: str
    published_at: datetime
    source: str
    summary: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def url_hash(self) -> str:
        return hashlib.sha256(self.url.encode("utf-8")).hexdigest()


class Fundamentals(BaseModel):
    """Point-in-time fundamental snapshot for a ticker."""

    ticker: str
    as_of: date
    market_cap: float | None = None
    pe_ratio: float | None = None
    revenue: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
