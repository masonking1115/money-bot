# Foundation & Data Layer Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the project scaffold and a cached, point-in-time price data layer that serves bars for a configured sector universe.

**Architecture:** A `src/`-layout Python package (`moneybot`) with pydantic-typed domain models, a YAML-driven sector universe + environment settings, an on-disk cache (SQLite metadata + parquet frames), and a `PriceProvider` protocol implemented by a yfinance-backed provider. A `DataLayer` facade ties universe + cache + provider together and enforces point-in-time (`as_of`) access so no future data can leak into research or backtests.

**Tech Stack:** Python 3.11+, uv (env/deps), pytest, pydantic v2, pydantic-settings, PyYAML, pandas, pyarrow, yfinance. Lint with ruff.

---

## File Structure

- `pyproject.toml` — project metadata, deps, pytest/ruff config
- `src/moneybot/__init__.py` — package marker + version
- `src/moneybot/models.py` — pydantic domain models (`Bar`, `Filing`, `NewsItem`, `Fundamentals`)
- `src/moneybot/config.py` — `Settings` (env) + `Universe`/`TickerMeta` (YAML) loaders
- `src/moneybot/cache.py` — `Cache`: SQLite kv + parquet dataframe store, content-hash helpers
- `src/moneybot/providers/__init__.py` — provider protocols (`PriceProvider`)
- `src/moneybot/providers/yfinance_price.py` — `YFinancePriceProvider`
- `src/moneybot/data_layer.py` — `DataLayer` facade (universe + cache + provider, point-in-time)
- `universe.example.yaml` — example sector universe config
- `.env.example` — example settings
- `tests/...` — mirrors the package; network is always mocked

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/moneybot/__init__.py`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "moneybot"
version = "0.1.0"
description = "AI sector-specialist trading bot"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "pyyaml>=6.0",
    "pandas>=2.2",
    "pyarrow>=15.0",
    "yfinance>=0.2.40",
]

[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.4"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/moneybot"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]
```

- [ ] **Step 2: Create `src/moneybot/__init__.py`**

```python
"""AI sector-specialist trading bot."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Write the smoke test `tests/test_smoke.py`**

```python
import moneybot


def test_package_imports_and_has_version():
    assert moneybot.__version__ == "0.1.0"
```

- [ ] **Step 4: Create the environment and install**

Run: `uv sync`
Expected: a `.venv/` is created and dependencies install without error.

- [ ] **Step 5: Run the smoke test to verify it passes**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: PASS (1 passed)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/moneybot/__init__.py tests/test_smoke.py
git commit -m "chore: project scaffold with pytest"
```

---

## Task 2: Domain models

**Files:**
- Create: `src/moneybot/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test `tests/test_models.py`**

```python
from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from moneybot.models import Bar, Filing, Fundamentals, NewsItem


def test_bar_roundtrips_fields():
    bar = Bar(
        ts=datetime(2026, 6, 9, 14, 0, tzinfo=timezone.utc),
        open=10.0, high=11.0, low=9.5, close=10.5, volume=1000,
    )
    assert bar.close == 10.5
    assert bar.volume == 1000


def test_bar_rejects_negative_volume():
    with pytest.raises(ValidationError):
        Bar(
            ts=datetime(2026, 6, 9, tzinfo=timezone.utc),
            open=1, high=1, low=1, close=1, volume=-5,
        )


def test_filing_computes_stable_content_hash():
    f1 = Filing(ticker="NVDA", form_type="10-K", filed_at=date(2026, 2, 1),
                accession_no="0001-26-000001", url="https://x/1", raw_text="hello")
    f2 = Filing(ticker="NVDA", form_type="10-K", filed_at=date(2026, 2, 1),
                accession_no="0001-26-000001", url="https://x/1", raw_text="hello")
    assert f1.content_hash == f2.content_hash
    assert len(f1.content_hash) == 64  # sha256 hex


def test_filing_hash_changes_with_text():
    base = dict(ticker="NVDA", form_type="8-K", filed_at=date(2026, 2, 1),
                accession_no="a", url="https://x")
    assert Filing(**base, raw_text="a").content_hash != Filing(**base, raw_text="b").content_hash


def test_newsitem_url_hash_is_deterministic():
    n = NewsItem(title="t", url="https://news/abc", published_at=datetime(2026, 6, 9, tzinfo=timezone.utc),
                 source="rss")
    assert n.url_hash == NewsItem(title="other", url="https://news/abc",
                                  published_at=datetime(2026, 6, 9, tzinfo=timezone.utc),
                                  source="rss").url_hash


def test_fundamentals_allows_optional_fields():
    fund = Fundamentals(ticker="AMD", as_of=date(2026, 6, 9))
    assert fund.market_cap is None
    assert fund.extra == {}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'moneybot.models'`

- [ ] **Step 3: Write `src/moneybot/models.py`**

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/models.py tests/test_models.py
git commit -m "feat: typed domain models (Bar, Filing, NewsItem, Fundamentals)"
```

---

## Task 3: Configuration (settings + universe)

**Files:**
- Create: `src/moneybot/config.py`
- Create: `universe.example.yaml`
- Create: `.env.example`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test `tests/test_config.py`**

```python
import textwrap

import pytest

from moneybot.config import Settings, Universe, load_universe


def test_settings_defaults_to_paper_mode(monkeypatch):
    monkeypatch.delenv("MONEYBOT_MODE", raising=False)
    settings = Settings()
    assert settings.mode == "paper"


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("MONEYBOT_MODE", "live")
    monkeypatch.setenv("MONEYBOT_DATA_DIR", "/tmp/mb-data")
    settings = Settings()
    assert settings.mode == "live"
    assert str(settings.data_dir) == "/tmp/mb-data"


def test_load_universe_parses_tickers_and_benchmark(tmp_path):
    path = tmp_path / "universe.yaml"
    path.write_text(textwrap.dedent("""
        sector: semiconductors
        benchmark: SMH
        tickers:
          - symbol: NVDA
            market_cap_tier: mega
            earnings_date: 2026-08-27
          - symbol: AMD
            market_cap_tier: large
    """))
    uni = load_universe(path)
    assert isinstance(uni, Universe)
    assert uni.sector == "semiconductors"
    assert uni.benchmark == "SMH"
    assert uni.symbols == ["NVDA", "AMD"]
    assert uni.get("NVDA").earnings_date.isoformat() == "2026-08-27"
    assert uni.get("AMD").earnings_date is None


def test_universe_get_unknown_symbol_raises(tmp_path):
    path = tmp_path / "u.yaml"
    path.write_text("sector: s\nbenchmark: B\ntickers:\n  - symbol: NVDA\n")
    uni = load_universe(path)
    with pytest.raises(KeyError):
        uni.get("TSLA")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'moneybot.config'`

- [ ] **Step 3: Write `src/moneybot/config.py`**

```python
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
    data_dir: Path = Path("data")
    cache_dir: Path = Path("cache")

    # API credentials (empty until provided; phase-1 providers degrade gracefully)
    alpaca_key_id: str = ""
    alpaca_secret_key: str = ""

    # Model tiering
    model_triage: str = "claude-haiku-4-5"
    model_deep_read: str = "claude-sonnet-4-6"
    model_analyst: str = "claude-opus-4-8"


class TickerMeta(BaseModel):
    symbol: str
    market_cap_tier: str | None = None
    earnings_date: date | None = None


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
```

- [ ] **Step 4: Create `universe.example.yaml`**

```yaml
# Example sector universe. Copy to universe.yaml and edit.
# Lean toward a sector where you have a real informational edge.
sector: semiconductors
benchmark: SMH
tickers:
  - symbol: NVDA
    market_cap_tier: mega
    earnings_date: 2026-08-27   # manual override; verify each quarter
  - symbol: AMD
    market_cap_tier: large
  - symbol: AVGO
    market_cap_tier: mega
```

- [ ] **Step 5: Create `.env.example`**

```bash
MONEYBOT_MODE=paper
MONEYBOT_DATA_DIR=data
MONEYBOT_CACHE_DIR=cache
MONEYBOT_ALPACA_KEY_ID=
MONEYBOT_ALPACA_SECRET_KEY=
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (4 passed)

- [ ] **Step 7: Commit**

```bash
git add src/moneybot/config.py universe.example.yaml .env.example tests/test_config.py
git commit -m "feat: settings and sector universe config"
```

---

## Task 4: On-disk cache

**Files:**
- Create: `src/moneybot/cache.py`
- Test: `tests/test_cache.py`

- [ ] **Step 1: Write the failing test `tests/test_cache.py`**

```python
import pandas as pd

from moneybot.cache import Cache


def test_json_set_get_roundtrip(tmp_path):
    cache = Cache(tmp_path)
    cache.set_json("k1", {"a": 1, "b": [2, 3]})
    assert cache.get_json("k1") == {"a": 1, "b": [2, 3]}


def test_json_get_missing_returns_none(tmp_path):
    cache = Cache(tmp_path)
    assert cache.get_json("nope") is None


def test_dataframe_set_get_roundtrip(tmp_path):
    cache = Cache(tmp_path)
    df = pd.DataFrame({"close": [1.0, 2.0], "volume": [10, 20]})
    cache.set_dataframe("bars:NVDA", df)
    out = cache.get_dataframe("bars:NVDA")
    pd.testing.assert_frame_equal(out, df)


def test_dataframe_get_missing_returns_none(tmp_path):
    cache = Cache(tmp_path)
    assert cache.get_dataframe("missing") is None


def test_persists_across_instances(tmp_path):
    Cache(tmp_path).set_json("persist", {"x": 1})
    assert Cache(tmp_path).get_json("persist") == {"x": 1}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'moneybot.cache'`

- [ ] **Step 3: Write `src/moneybot/cache.py`**

```python
"""On-disk cache: SQLite for small JSON values, parquet files for DataFrames."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


class Cache:
    """A simple keyed cache. JSON values live in SQLite; DataFrames as parquet."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.frames_dir = self.root / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.root / "cache.sqlite"
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT)")
        self._conn.commit()

    def set_json(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
            (key, json.dumps(value, sort_keys=True)),
        )
        self._conn.commit()

    def get_json(self, key: str) -> Any | None:
        row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _frame_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.frames_dir / f"{digest}.parquet"

    def set_dataframe(self, key: str, df: pd.DataFrame) -> None:
        df.to_parquet(self._frame_path(key), index=False)

    def get_dataframe(self, key: str) -> pd.DataFrame | None:
        path = self._frame_path(key)
        return pd.read_parquet(path) if path.exists() else None
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_cache.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/cache.py tests/test_cache.py
git commit -m "feat: on-disk cache (sqlite kv + parquet frames)"
```

---

## Task 5: Price provider protocol + yfinance implementation

**Files:**
- Create: `src/moneybot/providers/__init__.py`
- Create: `src/moneybot/providers/yfinance_price.py`
- Test: `tests/providers/test_yfinance_price.py`

- [ ] **Step 1: Write the provider protocol `src/moneybot/providers/__init__.py`**

```python
"""Data provider protocols. Implementations are swappable behind these interfaces."""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class PriceProvider(Protocol):
    def get_bars(
        self,
        ticker: str,
        timeframe: str,
        lookback: int,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        """Return OHLCV bars with a tz-aware 'ts' column, oldest first.

        If as_of is given, no bar with ts.date() > as_of may be returned
        (point-in-time discipline — prevents lookahead).
        """
        ...
```

- [ ] **Step 2: Write the failing test `tests/providers/test_yfinance_price.py`**

Create `tests/providers/__init__.py` (empty) first, then:

```python
from datetime import date, datetime, timezone

import pandas as pd

from moneybot.providers import PriceProvider
from moneybot.providers.yfinance_price import YFinancePriceProvider


def _fake_yf_frame():
    idx = pd.DatetimeIndex(
        [datetime(2026, 6, 7, tzinfo=timezone.utc),
         datetime(2026, 6, 8, tzinfo=timezone.utc),
         datetime(2026, 6, 9, tzinfo=timezone.utc)],
        name="Datetime",
    )
    return pd.DataFrame(
        {"Open": [1, 2, 3], "High": [1, 2, 3], "Low": [1, 2, 3],
         "Close": [10.0, 11.0, 12.0], "Volume": [100, 200, 300]},
        index=idx,
    )


def test_satisfies_protocol():
    assert isinstance(YFinancePriceProvider(), PriceProvider)


def test_get_bars_normalizes_columns(monkeypatch):
    prov = YFinancePriceProvider()
    monkeypatch.setattr(prov, "_download", lambda *a, **k: _fake_yf_frame())
    df = prov.get_bars("NVDA", timeframe="1d", lookback=5)
    assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert df["close"].tolist() == [10.0, 11.0, 12.0]
    assert df["ts"].is_monotonic_increasing


def test_as_of_filters_future_bars(monkeypatch):
    prov = YFinancePriceProvider()
    monkeypatch.setattr(prov, "_download", lambda *a, **k: _fake_yf_frame())
    df = prov.get_bars("NVDA", timeframe="1d", lookback=5, as_of=date(2026, 6, 8))
    assert df["ts"].dt.date.max() == date(2026, 6, 8)
    assert len(df) == 2
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/providers/test_yfinance_price.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'moneybot.providers.yfinance_price'`

- [ ] **Step 4: Write `src/moneybot/providers/yfinance_price.py`**

```python
"""yfinance-backed price provider (free, phase-1)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf


class YFinancePriceProvider:
    """Fetches OHLCV bars from yfinance and normalizes them to the house schema."""

    def _download(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        # Isolated for testing — patched in unit tests so no network is hit.
        return yf.download(
            ticker, period=period, interval=interval,
            auto_adjust=False, progress=False,
        )

    def get_bars(
        self,
        ticker: str,
        timeframe: str,
        lookback: int,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        period = f"{max(lookback, 1)}d" if timeframe.endswith("d") else f"{max(lookback, 1)}d"
        raw = self._download(ticker, period=period, interval=timeframe)
        if raw.empty:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

        # yfinance can return a column MultiIndex for single tickers; flatten it.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw.reset_index().rename(
            columns={
                raw.index.name or "Date": "ts",
                "Datetime": "ts",
                "Date": "ts",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df = df[["ts", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("ts").reset_index(drop=True)

        if as_of is not None:
            df = df[df["ts"].dt.date <= as_of].reset_index(drop=True)
        return df
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/providers/test_yfinance_price.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add src/moneybot/providers/__init__.py src/moneybot/providers/yfinance_price.py tests/providers/__init__.py tests/providers/test_yfinance_price.py
git commit -m "feat: PriceProvider protocol + yfinance implementation with point-in-time"
```

---

## Task 6: DataLayer facade (universe + cache + provider, point-in-time)

**Files:**
- Create: `src/moneybot/data_layer.py`
- Test: `tests/test_data_layer.py`

- [ ] **Step 1: Write the failing test `tests/test_data_layer.py`**

```python
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from moneybot.cache import Cache
from moneybot.config import Universe
from moneybot.data_layer import DataLayer


class StubPriceProvider:
    def __init__(self):
        self.calls = 0

    def get_bars(self, ticker, timeframe, lookback, as_of=None):
        self.calls += 1
        df = pd.DataFrame({
            "ts": pd.to_datetime(
                ["2026-06-07", "2026-06-08", "2026-06-09"], utc=True
            ),
            "open": [1, 2, 3], "high": [1, 2, 3], "low": [1, 2, 3],
            "close": [10.0, 11.0, 12.0], "volume": [100, 200, 300],
        })
        if as_of is not None:
            df = df[df["ts"].dt.date <= as_of].reset_index(drop=True)
        return df


def _universe():
    return Universe(sector="semis", benchmark="SMH",
                    tickers=[{"symbol": "NVDA"}, {"symbol": "AMD"}])


def test_get_bars_returns_provider_data(tmp_path):
    dl = DataLayer(_universe(), StubPriceProvider(), Cache(tmp_path))
    df = dl.get_bars("NVDA", "1d", 5)
    assert df["close"].tolist() == [10.0, 11.0, 12.0]


def test_rejects_ticker_outside_universe(tmp_path):
    dl = DataLayer(_universe(), StubPriceProvider(), Cache(tmp_path))
    with pytest.raises(ValueError, match="not in universe"):
        dl.get_bars("TSLA", "1d", 5)


def test_caches_live_bars_and_avoids_refetch(tmp_path):
    provider = StubPriceProvider()
    dl = DataLayer(_universe(), provider, Cache(tmp_path))
    dl.get_bars("NVDA", "1d", 5)
    dl.get_bars("NVDA", "1d", 5)
    assert provider.calls == 1  # second call served from cache


def test_point_in_time_requests_are_not_cached_and_filter(tmp_path):
    provider = StubPriceProvider()
    dl = DataLayer(_universe(), provider, Cache(tmp_path))
    df = dl.get_bars("NVDA", "1d", 5, as_of=date(2026, 6, 8))
    assert df["ts"].dt.date.max() == date(2026, 6, 8)
    # as_of requests always re-fetch (backtest correctness over cache reuse)
    dl.get_bars("NVDA", "1d", 5, as_of=date(2026, 6, 8))
    assert provider.calls == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_data_layer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'moneybot.data_layer'`

- [ ] **Step 3: Write `src/moneybot/data_layer.py`**

```python
"""DataLayer facade: bounds access to the universe, caches live data,
and enforces point-in-time access for backtests."""

from __future__ import annotations

from datetime import date

import pandas as pd

from moneybot.cache import Cache
from moneybot.config import Universe
from moneybot.providers import PriceProvider


class DataLayer:
    def __init__(self, universe: Universe, price_provider: PriceProvider, cache: Cache) -> None:
        self.universe = universe
        self.prices = price_provider
        self.cache = cache

    def _require_in_universe(self, ticker: str) -> None:
        if ticker not in self.universe.symbols and ticker != self.universe.benchmark:
            raise ValueError(f"{ticker} not in universe")

    def get_bars(
        self,
        ticker: str,
        timeframe: str,
        lookback: int,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        self._require_in_universe(ticker)

        # Point-in-time requests bypass the cache: backtest correctness beats reuse,
        # and a cached "live" frame may contain bars newer than as_of.
        if as_of is not None:
            return self.prices.get_bars(ticker, timeframe, lookback, as_of=as_of)

        key = f"bars:{ticker}:{timeframe}:{lookback}"
        cached = self.cache.get_dataframe(key)
        if cached is not None:
            return cached

        df = self.prices.get_bars(ticker, timeframe, lookback)
        self.cache.set_dataframe(key, df)
        return df
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_data_layer.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Run the full suite + lint**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all tests pass; ruff reports no errors.

- [ ] **Step 6: Commit**

```bash
git add src/moneybot/data_layer.py tests/test_data_layer.py
git commit -m "feat: DataLayer facade with universe bounding, caching, point-in-time"
```

---

## Task 7: README quickstart + push

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

````markdown
# money-bot

AI sector-specialist trading bot. Agents interpret unstructured data (filings, news);
deterministic code handles risk and execution. Paper-first; live by a single config flag.

See the design spec in `docs/superpowers/specs/` and plans in `docs/superpowers/plans/`.

## Setup

```bash
uv sync
cp .env.example .env            # fill in as you add providers
cp universe.example.yaml universe.yaml   # edit to your sector
uv run pytest -q
```

## Status

Phase 1 (this plan): foundation + point-in-time price data layer.
````

- [ ] **Step 2: Commit and push**

```bash
git add README.md
git commit -m "docs: README quickstart"
git push
```

Expected: push succeeds to `origin/main`.

---

## Self-Review Notes

- **Spec coverage (Plan 1 scope):** pluggable provider protocol ✓ (Task 5), caching ✓ (Task 4), bounded universe config ✓ (Task 3), point-in-time discipline ✓ (Tasks 5–6), normalized typed models ✓ (Task 2), paper/live mode flag in settings ✓ (Task 3). Filings/news/fundamentals providers, broker price feed, agents, risk engine, memory, and validation are intentionally deferred to later plans (see plan roadmap).
- **Type consistency:** `Universe.symbols`/`.get()`, `PriceProvider.get_bars(...)`, `Cache.get_dataframe/set_dataframe/get_json/set_json`, and `DataLayer.get_bars(...)` are used identically across tasks.
- **No placeholders:** every code step contains complete, runnable code and exact commands.
