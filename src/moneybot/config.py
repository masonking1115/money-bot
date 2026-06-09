"""Environment settings and the sector universe configuration."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process settings, sourced from environment (prefix MONEYBOT_) and .env."""

    model_config = SettingsConfigDict(env_prefix="MONEYBOT_", env_file=".env", extra="ignore")

    mode: Literal["paper", "live"] = "paper"
    data_dir: str = "data"
    cache_dir: str = "cache"

    # API credentials (empty until provided; phase-1 providers degrade gracefully)
    alpaca_key_id: str = ""
    alpaca_secret_key: str = ""

    sec_user_agent: str = "moneybot mason@voltai.com"

    # Model tiering
    model_triage: str = "claude-haiku-4-5"
    model_deep_read: str = "claude-sonnet-4-6"
    model_analyst: str = "claude-opus-4-8"

    # Active strategy (resolved via moneybot.strategies.registry)
    strategy: str = "catalyst_driven"


class TickerMeta(BaseModel):
    symbol: str
    market_cap_tier: str | None = None
    earnings_date: date | None = None
    cik: str | None = None


class Universe(BaseModel):
    sector: str
    benchmark: str
    tickers: list[TickerMeta]

    @property
    def symbols(self) -> list[str]:
        return [t.symbol for t in self.tickers]

    def get(self, symbol: str) -> TickerMeta:
        for t in self.tickers:
            if t.symbol == symbol:
                return t
        raise KeyError(f"{symbol} not in universe")


def load_universe(path: str | Path) -> Universe:
    data = yaml.safe_load(Path(path).read_text())
    return Universe.model_validate(data)
