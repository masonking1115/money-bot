# Backtesting Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replay historical market data day-by-day through the *existing* Orchestrator/Research/Analyst/Risk/Execution code path (point-in-time, no lookahead) and produce a performance report so the operator can tune Risk Engine parameters and validate the strategy edge before paper→live.

**Architecture:** A new `src/moneybot/backtest/` package that *drives* `Orchestrator.run_cycle(as_of=day)` across each historical trading day. A `SimClock` feeds the orchestrator the simulated date (so `cycle_id` and the market-hours gate reflect history); `market_open` is forced open and the trading calendar comes from the benchmark's real bar dates. The expensive AI step is made replay-cheap by caching **agent outputs keyed by date** (research signals + analyst plans), and prices are cached by `(ticker, timeframe, lookback, as_of)`. The first run pays the LLM + network cost once; every subsequent risk-parameter sweep replays from cache — zero LLM calls, fully offline. After each daily cycle the harness marks the portfolio to market (reusing `build_portfolio_state`) to record an equity point, then computes return / max-drawdown / Sharpe / win-rate and compares against buy-and-hold SMH.

**Tech Stack:** Python 3.12, uv toolchain (`uv run pytest`, `uv run ruff check`), pydantic v2, pandas, pytest + `tmp_path`. Production modules use `from __future__ import annotations`; test files omit it. Ruff line-length 100 (self-check — not enforced by the default rule set). No test may hit the network or an LLM; no fabricated clock inside components (inject the clock).

---

## Design Notes (read before starting)

**Why cache agent outputs (not raw LLM calls):** `Orchestrator.run_cycle` calls exactly two methods on the AI layer — `research.research_universe(as_of=...) -> dict[str, list[CatalystSignal]]` and `analyst.analyze(research, as_of=...) -> list[TradePlan]` (see `src/moneybot/orchestrator/engine.py:79-80`). Both outputs depend only on the simulated date + universe + agent configuration — **not** on Risk Engine parameters or current positions. Caching them keyed by `as_of` date means a risk/exit-parameter sweep replays the frozen AI decisions instantly and deterministically. (Caching at the raw-LLM level would be fragile: free news providers are not perfectly reproducible, so prompt text can drift and miss the cache. Date-keyed output caching sidesteps that.)

**Cache validity assumption:** the agent-output cache is keyed by date only. It is valid as long as the **research/analyst configuration is unchanged** (universe, `analyst_shortlist`, `rs_lookback_days`, `rs_timeframe`, strategy ranking). Tuning Risk Engine / exit parameters (sizing, stops, caps, `daily_loss_limit_pct`, etc.) is safe against the cache. If the operator changes research/analyst config, they must re-record (delete the cache dir or pass `--record`). This is documented in the CLI help and README.

**Equity curve = mark-to-market.** `PaperBroker.get_account()` is marked-to-*cost* (`src/moneybot/execution/paper.py`), so it does not reflect unrealized price moves. The harness records the equity curve by calling the existing point-in-time `build_portfolio_state(..., as_of=day)` after each cycle and reading `.equity` (which marks open positions to that day's close). Realized + unrealized P&L is therefore captured.

**Trade log from Fills, not the journal.** Each `CycleResult` carries `entry_fills` and `exit_fills` (`list[Fill]`, each with `avg_price`, `filled_qty`, `ticker`, `side`). The harness accumulates all fills across cycles and pairs buys→sells FIFO per ticker to compute per-trade realized P&L and win rate. (The journal's `exit` entries do not store the exit price, so Fills are the right source.)

**Daily-breaker limitation (document, do not fix here):** with one cycle per simulated day, `SodEquityStore` re-anchors each day, so `day_pnl_pct ≈ 0` at every cycle and the intraday daily-loss circuit breaker never fires in a daily backtest. This is inherent to daily cadence (the breaker is an intraday protection). Note it in the report header and README; it is not a bug.

**Network discipline:** the harness *itself*, when the operator runs it (`python -m moneybot.backtest ...`), legitimately hits yfinance / EDGAR / Anthropic on a record run — it is an operator tool, not the test suite. Every *test* in this plan uses fakes/stubs and `tmp_path` and must not touch the network or an LLM.

---

## File Structure

- Modify: `src/moneybot/orchestrator/factory.py` — add optional `research` / `analyst` injection params.
- Create: `src/moneybot/backtest/__init__.py` — package exports.
- Create: `src/moneybot/backtest/models.py` — `BacktestConfig`, `EquityPoint`, `TradeRecord`, `PerformanceMetrics`, `BacktestReport`.
- Create: `src/moneybot/backtest/clock.py` — `SimClock` (mutable, injected as the orchestrator clock).
- Create: `src/moneybot/backtest/calendar.py` — `trading_days_from_bars`.
- Create: `src/moneybot/backtest/price_cache.py` — `CachingPriceProvider` (record/replay, DataFrame↔JSON).
- Create: `src/moneybot/backtest/agent_cache.py` — `CachingResearch`, `CachingAnalyst` (record/replay, date-keyed).
- Create: `src/moneybot/backtest/metrics.py` — pure metrics + trade-log builder + benchmark buy-and-hold.
- Create: `src/moneybot/backtest/engine.py` — `run_backtest(...)` driver.
- Create: `src/moneybot/backtest/report.py` — render markdown summary + CSV/JSON export.
- Create: `src/moneybot/backtest/__main__.py` — CLI composition root (real providers + caches → run → print).
- Tests under `tests/backtest/` mirroring each module.
- Modify: `README.md` — add a "Phase 10: Backtesting" prose section.

---

### Task 1: Inject pre-built research/analyst into `build_orchestrator`

The backtest wraps the real research/analyst agents in caching decorators and must hand them to the orchestrator. `build_orchestrator` currently builds them internally; add optional params that, when provided, are used instead of building from the LLM (mirrors the existing `llm`/`clock`/`market_open` injection philosophy). Backward-compatible — production callers pass nothing and behavior is unchanged.

**Files:**
- Modify: `src/moneybot/orchestrator/factory.py:33-54`
- Test: `tests/orchestrator/test_factory.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/orchestrator/test_factory.py` (do NOT add `from __future__ import annotations`). Reuse the file's existing fakes if present; if the file already defines `FakeData`/`FakeLLM`/`StubRetriever`, reuse them and only add the test function below. If it does not, add these minimal fakes too.

```python
def test_build_orchestrator_uses_injected_research_and_analyst():
    # When research/analyst are supplied, the factory must use them verbatim
    # (not build its own from the LLM) — this is the backtest's injection seam.
    from moneybot.orchestrator.factory import build_orchestrator

    sentinel_research = object()
    sentinel_analyst = object()

    orch = build_orchestrator(
        settings=_settings(),               # existing helper in this test file
        data_layer=FakeData(),              # existing fake in this test file
        retriever=StubRetriever(),          # existing fake in this test file
        llm=FakeLLM(),                       # existing fake in this test file
        research=sentinel_research,
        analyst=sentinel_analyst,
    )

    assert orch.research is sentinel_research
    assert orch.analyst is sentinel_analyst
```

If `_settings()`, `FakeData`, `StubRetriever`, or `FakeLLM` do not already exist in the file, read the file first and adapt the names to whatever fakes/helpers it already provides. Do not duplicate fakes under new names.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/orchestrator/test_factory.py::test_build_orchestrator_uses_injected_research_and_analyst -v`
Expected: FAIL — `build_orchestrator()` got an unexpected keyword argument `research`.

- [ ] **Step 3: Add the optional params**

In `src/moneybot/orchestrator/factory.py`, change the signature and the two build lines:

```python
def build_orchestrator(
    *,
    settings: Settings,
    data_layer: DataLayer,
    retriever: MemoryRetriever,
    llm: LLMClient | None = None,
    clock: Callable[[], datetime] | None = None,
    market_open: Callable[[datetime], bool] = is_market_open,
    research=None,
    analyst=None,
) -> Orchestrator:
    clock = clock or (lambda: datetime.now(timezone.utc))

    if research is None:
        research = build_research_agent(
            settings=settings, data_layer=data_layer, retriever=retriever, llm=llm
        )
    if analyst is None:
        analyst = build_analyst_agent(
            settings=settings, data_layer=data_layer, retriever=retriever, llm=llm
        )
    risk = build_risk_engine(settings=settings, data_layer=data_layer)
    # ... rest unchanged ...
```

Add a one-line docstring note to the existing module/function docstring: "research/analyst may be injected pre-built (the backtest harness supplies caching wrappers)."

- [ ] **Step 4: Run test to verify it passes (and nothing regressed)**

Run: `uv run pytest tests/orchestrator/test_factory.py -v`
Expected: PASS (new test + all existing factory tests).

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/orchestrator/factory.py tests/orchestrator/test_factory.py
git commit -m "feat(orchestrator): allow injecting pre-built research/analyst into build_orchestrator"
```

---

### Task 2: Backtest data models

Typed config, per-day equity point, per-trade record, computed metrics, and the final report container.

**Files:**
- Create: `src/moneybot/backtest/__init__.py`
- Create: `src/moneybot/backtest/models.py`
- Test: `tests/backtest/test_models.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/__init__.py` (empty) and `tests/backtest/test_models.py`:

```python
from datetime import date

from moneybot.backtest.models import (
    BacktestConfig,
    BacktestReport,
    EquityPoint,
    PerformanceMetrics,
    TradeRecord,
)


def test_config_defaults():
    cfg = BacktestConfig(start=date(2024, 1, 1), end=date(2024, 6, 30))
    assert cfg.timeframe == "1d"
    assert cfg.starting_cash == 100_000.0
    assert cfg.mode == "record"
    assert cfg.use_agents is True


def test_config_rejects_end_before_start():
    import pytest

    with pytest.raises(ValueError):
        BacktestConfig(start=date(2024, 6, 30), end=date(2024, 1, 1))


def test_report_round_trips():
    report = BacktestReport(
        config=BacktestConfig(start=date(2024, 1, 1), end=date(2024, 1, 31)),
        equity_curve=[EquityPoint(day=date(2024, 1, 2), equity=100_000.0, cash=100_000.0, n_positions=0)],
        trades=[
            TradeRecord(
                ticker="NVDA", qty=10, entry_date=date(2024, 1, 2), exit_date=date(2024, 1, 10),
                entry_price=100.0, exit_price=110.0, pnl=100.0, pnl_pct=0.10, exit_reason="profit_target",
            )
        ],
        metrics=PerformanceMetrics(
            total_return=0.10, cagr=0.5, max_drawdown=0.05, sharpe=1.2,
            win_rate=1.0, n_trades=1, final_equity=110_000.0,
            benchmark_return=0.04, benchmark_final_equity=104_000.0,
        ),
    )
    again = BacktestReport.model_validate_json(report.model_dump_json())
    assert again.metrics.total_return == 0.10
    assert again.trades[0].ticker == "NVDA"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backtest/test_models.py -v`
Expected: FAIL — module `moneybot.backtest.models` not found.

- [ ] **Step 3: Write the models**

Create `src/moneybot/backtest/__init__.py`:

```python
"""Backtesting harness: replay historical data through the live orchestrator code path."""
```

Create `src/moneybot/backtest/models.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backtest/test_models.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/backtest/__init__.py src/moneybot/backtest/models.py tests/backtest/
git commit -m "feat(backtest): typed config/equity/trade/metrics/report models"
```

---

### Task 3: SimClock

A mutable clock the harness advances to each simulated day. The orchestrator derives `cycle_id` and its market-hours check from it, so it must return a tz-aware datetime at a fixed intraday time (10:00 America/New_York) for the day being replayed.

**Files:**
- Create: `src/moneybot/backtest/clock.py`
- Test: `tests/backtest/test_clock.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_clock.py`:

```python
from datetime import date, datetime
from zoneinfo import ZoneInfo

from moneybot.backtest.clock import SimClock

ET = ZoneInfo("America/New_York")


def test_returns_fixed_intraday_time_for_set_day():
    clock = SimClock()
    clock.set_day(date(2024, 3, 1))
    now = clock()
    assert now == datetime(2024, 3, 1, 10, 0, tzinfo=ET)
    # cycle_id derivation (as the orchestrator does it) is unique per day
    assert now.strftime("%Y-%m-%dT%H") == "2024-03-01T10"


def test_advancing_changes_the_returned_day():
    clock = SimClock()
    clock.set_day(date(2024, 3, 1))
    clock.set_day(date(2024, 3, 4))
    assert clock().date() == date(2024, 3, 4)


def test_call_before_set_raises():
    import pytest

    with pytest.raises(RuntimeError):
        SimClock()()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backtest/test_clock.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the SimClock**

Create `src/moneybot/backtest/clock.py`:

```python
"""A mutable clock for replay. The harness advances it to each simulated day;
the orchestrator reads it for cycle_id + market-hours gating. The fixed 10:00 ET
intraday time keeps cycle_id unique per day and inside normal market hours."""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_CYCLE_TIME = time(10, 0)


class SimClock:
    def __init__(self) -> None:
        self._day: date | None = None

    def set_day(self, day: date) -> None:
        self._day = day

    def __call__(self) -> datetime:
        if self._day is None:
            raise RuntimeError("SimClock used before set_day()")
        return datetime.combine(self._day, _CYCLE_TIME, tzinfo=_ET)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backtest/test_clock.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/backtest/clock.py tests/backtest/test_clock.py
git commit -m "feat(backtest): SimClock for date-driven replay"
```

---

### Task 4: Trading calendar from benchmark bars

The set of simulated days = the benchmark's (SMH) actual daily bar dates within `[start, end]`. Using real bar dates excludes weekends and market holidays automatically — no hand-maintained holiday list.

**Files:**
- Create: `src/moneybot/backtest/calendar.py`
- Test: `tests/backtest/test_calendar.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_calendar.py`:

```python
from datetime import date

import pandas as pd

from moneybot.backtest.calendar import trading_days_from_bars


def _bars(days):
    ts = pd.to_datetime([f"{d} 00:00:00+00:00" for d in days])
    return pd.DataFrame({"ts": ts, "close": [1.0] * len(days)})


def test_filters_to_range_and_sorts():
    bars = _bars(["2024-01-02", "2024-01-05", "2024-01-03", "2023-12-29", "2024-02-01"])
    out = trading_days_from_bars(bars, date(2024, 1, 1), date(2024, 1, 31))
    assert out == [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5)]


def test_empty_bars_yields_no_days():
    assert trading_days_from_bars(pd.DataFrame({"ts": [], "close": []}), date(2024, 1, 1), date(2024, 1, 31)) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backtest/test_calendar.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the calendar helper**

Create `src/moneybot/backtest/calendar.py`:

```python
"""Derive the replay trading calendar from the benchmark's real bar dates."""

from __future__ import annotations

from datetime import date

import pandas as pd


def trading_days_from_bars(bars: pd.DataFrame, start: date, end: date) -> list[date]:
    """Sorted, de-duplicated dates from `bars['ts']` that fall within [start, end]."""
    if bars.empty:
        return []
    days = sorted({ts.date() for ts in pd.to_datetime(bars["ts"])})
    return [d for d in days if start <= d <= end]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backtest/test_calendar.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/backtest/calendar.py tests/backtest/test_calendar.py
git commit -m "feat(backtest): trading calendar from benchmark bar dates"
```

---

### Task 5: CachingPriceProvider

A `PriceProvider` decorator that persists bars by `(ticker, timeframe, lookback, as_of)` so replay runs are offline and fast. In `record` mode it calls the inner provider on a miss and stores the result; in `replay` mode a miss raises (the cache must already be populated). Conforms to the `PriceProvider` protocol (`src/moneybot/providers/__init__.py:13-27`).

**Files:**
- Create: `src/moneybot/backtest/price_cache.py`
- Test: `tests/backtest/test_price_cache.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_price_cache.py`:

```python
from datetime import date

import pandas as pd
import pytest

from moneybot.backtest.price_cache import CachingPriceProvider


class CountingProvider:
    def __init__(self):
        self.calls = 0

    def get_bars(self, ticker, timeframe, lookback, as_of=None):
        self.calls += 1
        ts = pd.to_datetime([f"{as_of} 00:00:00+00:00"])
        return pd.DataFrame({"ts": ts, "open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100]})


def test_record_then_hit_does_not_recall_inner(tmp_path):
    inner = CountingProvider()
    cache = CachingPriceProvider(inner, root=tmp_path, mode="record")
    a = cache.get_bars("NVDA", "1d", 20, as_of=date(2024, 3, 1))
    b = cache.get_bars("NVDA", "1d", 20, as_of=date(2024, 3, 1))
    assert inner.calls == 1                      # second call served from cache
    assert list(a["close"]) == list(b["close"])
    assert str(b["ts"].dt.date.iloc[0]) == "2024-03-01"  # ts round-trips as tz-aware


def test_cache_key_separates_args(tmp_path):
    inner = CountingProvider()
    cache = CachingPriceProvider(inner, root=tmp_path, mode="record")
    cache.get_bars("NVDA", "1d", 20, as_of=date(2024, 3, 1))
    cache.get_bars("NVDA", "1d", 20, as_of=date(2024, 3, 2))  # different as_of
    cache.get_bars("AMD", "1d", 20, as_of=date(2024, 3, 1))   # different ticker
    assert inner.calls == 3


def test_replay_miss_raises(tmp_path):
    inner = CountingProvider()
    cache = CachingPriceProvider(inner, root=tmp_path, mode="replay")
    with pytest.raises(RuntimeError, match="cache miss"):
        cache.get_bars("NVDA", "1d", 20, as_of=date(2024, 3, 1))
    assert inner.calls == 0


def test_persists_across_instances(tmp_path):
    CachingPriceProvider(CountingProvider(), root=tmp_path, mode="record").get_bars(
        "NVDA", "1d", 20, as_of=date(2024, 3, 1)
    )
    reopened = CachingPriceProvider(CountingProvider(), root=tmp_path, mode="replay")
    df = reopened.get_bars("NVDA", "1d", 20, as_of=date(2024, 3, 1))
    assert df["close"].iloc[0] == 1.5  # read back from disk, inner never called
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backtest/test_price_cache.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the CachingPriceProvider**

Create `src/moneybot/backtest/price_cache.py`:

```python
"""Persisted point-in-time price cache, keyed by (ticker, timeframe, lookback, as_of).

Wraps any PriceProvider. record mode populates on miss; replay mode requires a hit.
Bars are stored as JSON (ts serialized as ISO-8601 with offset) so reads reconstruct
a tz-aware 'ts' column identical to the live provider's contract."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd

from moneybot.providers import PriceProvider

_BAR_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]


class CachingPriceProvider:
    def __init__(self, inner: PriceProvider, *, root: str | Path, mode: Literal["record", "replay"]) -> None:
        self._inner = inner
        self._dir = Path(root) / "prices"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._mode = mode

    def _path(self, ticker: str, timeframe: str, lookback: int, as_of: date | None) -> Path:
        key = f"{ticker}__{timeframe}__{lookback}__{as_of}"
        return self._dir / f"{key}.json"

    def get_bars(
        self, ticker: str, timeframe: str, lookback: int, as_of: date | None = None
    ) -> pd.DataFrame:
        path = self._path(ticker, timeframe, lookback, as_of)
        if path.exists():
            records = json.loads(path.read_text(encoding="utf-8"))
            df = pd.DataFrame(records, columns=_BAR_COLUMNS)
            if not df.empty:
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
            return df
        if self._mode == "replay":
            raise RuntimeError(
                f"cache miss in replay mode for {ticker} {timeframe} {lookback} as_of={as_of}; "
                "run a record pass first"
            )
        df = self._inner.get_bars(ticker, timeframe, lookback, as_of=as_of)
        out = df.copy()
        if not out.empty:
            out["ts"] = pd.to_datetime(out["ts"], utc=True).map(lambda t: t.isoformat())
        records = out.to_dict(orient="records")
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(records), encoding="utf-8")
        tmp.replace(path)
        return df
```

Note: store `ts` as ISO strings (`.isoformat()`) and reconstruct with `pd.to_datetime(..., utc=True)` so the reconstructed frame matches the provider contract (tz-aware `ts`, oldest first — order is preserved by `to_dict(orient="records")`). The atomic temp-then-replace write matches the codebase convention (`os.replace`/`Path.replace`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backtest/test_price_cache.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/backtest/price_cache.py tests/backtest/test_price_cache.py
git commit -m "feat(backtest): persisted point-in-time price cache"
```

---

### Task 6: CachingResearch + CachingAnalyst (date-keyed agent-output cache)

Decorators that cache the two AI outputs keyed by `as_of` date. `CachingResearch.research_universe(as_of)` caches `dict[str, list[CatalystSignal]]`; `CachingAnalyst.analyze(research, as_of)` caches `list[TradePlan]`. record mode calls the wrapped real agent on a miss and stores; replay mode returns the stored value (and raises on a miss). They duck-type the methods the orchestrator calls (`engine.py:79-80`).

**Files:**
- Create: `src/moneybot/backtest/agent_cache.py`
- Test: `tests/backtest/test_agent_cache.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_agent_cache.py`:

```python
from datetime import date

import pytest

from moneybot.analyst.models import TradePlan
from moneybot.backtest.agent_cache import CachingAnalyst, CachingResearch
from moneybot.strategies.models import CatalystSignal, Evidence, ExitPlan


def _signal(ticker):
    return CatalystSignal(
        ticker=ticker, category="demand", direction="bullish", materiality=0.8,
        freshness_days=1, conviction=0.7,
        evidence=[Evidence(source="8-K", quote="q", url="http://x")], thesis="t",
        signal_id="sig1",
    )


def _plan(ticker):
    return TradePlan(
        ticker=ticker, action="buy", conviction=0.6, thesis="t", score=1.0,
        signal_ref="sig1",
        exit_plan=ExitPlan(max_hold_days=10, stop_loss_pct=0.08, profit_target_pct=0.2, thesis_check_guidance="n/a"),
        analyst_note="ok",
    )


class FakeResearch:
    def __init__(self):
        self.calls = 0

    def research_universe(self, as_of=None):
        self.calls += 1
        return {"NVDA": [_signal("NVDA")]}


class FakeAnalyst:
    def __init__(self):
        self.calls = 0

    def analyze(self, research, as_of=None):
        self.calls += 1
        return [_plan("NVDA")]


def test_research_records_then_replays_without_recalling(tmp_path):
    inner = FakeResearch()
    cache = CachingResearch(inner, root=tmp_path, mode="record")
    a = cache.research_universe(as_of=date(2024, 3, 1))
    b = cache.research_universe(as_of=date(2024, 3, 1))
    assert inner.calls == 1
    assert isinstance(b["NVDA"][0], CatalystSignal)
    assert b["NVDA"][0].ticker == "NVDA"
    assert a["NVDA"][0].signal_id == "sig1"


def test_analyst_records_then_replays_without_recalling(tmp_path):
    inner = FakeAnalyst()
    cache = CachingAnalyst(inner, root=tmp_path, mode="record")
    cache.analyze({"NVDA": [_signal("NVDA")]}, as_of=date(2024, 3, 1))
    plans = cache.analyze({"NVDA": [_signal("NVDA")]}, as_of=date(2024, 3, 1))
    assert inner.calls == 1
    assert isinstance(plans[0], TradePlan)
    assert plans[0].ticker == "NVDA"


def test_distinct_days_recompute(tmp_path):
    inner = FakeResearch()
    cache = CachingResearch(inner, root=tmp_path, mode="record")
    cache.research_universe(as_of=date(2024, 3, 1))
    cache.research_universe(as_of=date(2024, 3, 2))
    assert inner.calls == 2


def test_replay_miss_raises(tmp_path):
    cache = CachingAnalyst(FakeAnalyst(), root=tmp_path, mode="replay")
    with pytest.raises(RuntimeError, match="cache miss"):
        cache.analyze({}, as_of=date(2024, 3, 1))


def test_research_persists_across_instances(tmp_path):
    CachingResearch(FakeResearch(), root=tmp_path, mode="record").research_universe(as_of=date(2024, 3, 1))
    inner = FakeResearch()
    reopened = CachingResearch(inner, root=tmp_path, mode="replay")
    out = reopened.research_universe(as_of=date(2024, 3, 1))
    assert inner.calls == 0
    assert out["NVDA"][0].ticker == "NVDA"


def test_as_of_none_uses_stable_key(tmp_path):
    inner = FakeResearch()
    cache = CachingResearch(inner, root=tmp_path, mode="record")
    cache.research_universe(as_of=None)
    cache.research_universe(as_of=None)
    assert inner.calls == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backtest/test_agent_cache.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the agent caches**

Create `src/moneybot/backtest/agent_cache.py`:

```python
"""Date-keyed caches for the two expensive AI outputs.

CachingResearch wraps research_universe(as_of) -> dict[ticker, list[CatalystSignal]].
CachingAnalyst wraps analyze(research, as_of) -> list[TradePlan].

Keyed by the simulated date only: the AI output depends on date + universe + agent
config, never on Risk Engine parameters, so a risk-parameter sweep replays the frozen
decisions for free. record mode populates on miss; replay mode requires a hit."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from moneybot.analyst.models import TradePlan
from moneybot.strategies.models import CatalystSignal

if TYPE_CHECKING:
    pass


def _key(as_of: date | None) -> str:
    return as_of.isoformat() if as_of is not None else "_none_"


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class CachingResearch:
    def __init__(self, inner, *, root: str | Path, mode: Literal["record", "replay"]) -> None:
        self._inner = inner
        self._dir = Path(root) / "research"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._mode = mode

    def research_universe(self, as_of: date | None = None) -> dict[str, list[CatalystSignal]]:
        path = self._dir / f"{_key(as_of)}.json"
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            return {
                ticker: [CatalystSignal.model_validate(s) for s in sigs]
                for ticker, sigs in raw.items()
            }
        if self._mode == "replay":
            raise RuntimeError(f"cache miss in replay mode for research as_of={as_of}")
        result = self._inner.research_universe(as_of=as_of)
        raw = {ticker: [s.model_dump(mode="json") for s in sigs] for ticker, sigs in result.items()}
        _atomic_write(path, json.dumps(raw))
        return result


class CachingAnalyst:
    def __init__(self, inner, *, root: str | Path, mode: Literal["record", "replay"]) -> None:
        self._inner = inner
        self._dir = Path(root) / "analyst"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._mode = mode

    def analyze(self, research: dict[str, list[CatalystSignal]], as_of: date | None = None) -> list[TradePlan]:
        path = self._dir / f"{_key(as_of)}.json"
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            return [TradePlan.model_validate(p) for p in raw]
        if self._mode == "replay":
            raise RuntimeError(f"cache miss in replay mode for analyst as_of={as_of}")
        plans = self._inner.analyze(research, as_of=as_of)
        _atomic_write(path, json.dumps([p.model_dump(mode="json") for p in plans]))
        return plans
```

Remove the empty `if TYPE_CHECKING: pass` block if ruff flags it; it is only a placeholder. Self-check ruff line length ≤ 100.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backtest/test_agent_cache.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/backtest/agent_cache.py tests/backtest/test_agent_cache.py
git commit -m "feat(backtest): date-keyed caches for research signals and analyst plans"
```

---

### Task 7: Performance metrics (pure)

Pure functions: build the trade log from accumulated Fills (FIFO buy→sell per ticker), and compute headline metrics from the equity curve + trade log + benchmark series. No I/O, no network — fully unit-testable.

**Files:**
- Create: `src/moneybot/backtest/metrics.py`
- Test: `tests/backtest/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_metrics.py`:

```python
from datetime import date, datetime, timezone

import pytest

from moneybot.backtest.metrics import (
    build_trade_log,
    compute_metrics,
    max_drawdown,
    sharpe,
)
from moneybot.backtest.models import EquityPoint
from moneybot.execution.models import Fill


def _fill(ticker, side, qty, price, day):
    return Fill(
        client_order_id=f"{ticker}:{side}:{day}", broker_order_id="b", ticker=ticker,
        side=side, status="filled", filled_qty=qty, avg_price=price,
        ts=datetime(day.year, day.month, day.day, tzinfo=timezone.utc),
    )


def test_max_drawdown_simple():
    # 100 -> 120 -> 90 -> 110 : worst peak-to-trough is 120 -> 90 = -25%
    eq = [100.0, 120.0, 90.0, 110.0]
    assert max_drawdown(eq) == pytest.approx(0.25)


def test_max_drawdown_monotonic_increase_is_zero():
    assert max_drawdown([100.0, 110.0, 120.0]) == 0.0


def test_sharpe_zero_when_no_variance():
    assert sharpe([0.01, 0.01, 0.01]) == 0.0  # std == 0 -> defined as 0


def test_sharpe_positive_for_positive_mean():
    assert sharpe([0.01, -0.005, 0.02, 0.0]) > 0


def test_build_trade_log_fifo_realized_pnl():
    fills = [
        _fill("NVDA", "buy", 10, 100.0, date(2024, 1, 2)),
        _fill("NVDA", "sell", 10, 110.0, date(2024, 1, 10)),
    ]
    trades = build_trade_log(fills)
    assert len(trades) == 1
    t = trades[0]
    assert t.ticker == "NVDA" and t.qty == 10
    assert t.entry_price == 100.0 and t.exit_price == 110.0
    assert t.pnl == pytest.approx(100.0)
    assert t.pnl_pct == pytest.approx(0.10)


def test_build_trade_log_partial_then_full_exit():
    fills = [
        _fill("AMD", "buy", 10, 50.0, date(2024, 1, 2)),
        _fill("AMD", "sell", 4, 55.0, date(2024, 1, 5)),
        _fill("AMD", "sell", 6, 45.0, date(2024, 1, 9)),
    ]
    trades = build_trade_log(fills)
    assert len(trades) == 2
    assert trades[0].qty == 4 and trades[0].pnl == pytest.approx(20.0)
    assert trades[1].qty == 6 and trades[1].pnl == pytest.approx(-30.0)


def test_build_trade_log_ignores_rejected_and_unmatched():
    fills = [
        _fill("NVDA", "buy", 10, 100.0, date(2024, 1, 2)),  # still open at end -> no trade
    ]
    fills[0] = fills[0].model_copy(update={"status": "rejected"})
    assert build_trade_log(fills) == []


def test_compute_metrics_end_to_end():
    curve = [
        EquityPoint(day=date(2024, 1, 2), equity=100_000.0, cash=0.0, n_positions=1),
        EquityPoint(day=date(2024, 1, 3), equity=110_000.0, cash=0.0, n_positions=1),
    ]
    trades = build_trade_log([
        _fill("NVDA", "buy", 10, 100.0, date(2024, 1, 2)),
        _fill("NVDA", "sell", 10, 110.0, date(2024, 1, 3)),
    ])
    m = compute_metrics(
        equity_curve=curve, trades=trades, starting_cash=100_000.0,
        benchmark_closes=[200.0, 204.0],
    )
    assert m.total_return == pytest.approx(0.10)
    assert m.final_equity == 110_000.0
    assert m.win_rate == 1.0
    assert m.n_trades == 1
    assert m.benchmark_return == pytest.approx(0.02)
    assert m.benchmark_final_equity == pytest.approx(102_000.0)


def test_compute_metrics_empty_curve_is_safe():
    m = compute_metrics(equity_curve=[], trades=[], starting_cash=100_000.0, benchmark_closes=[])
    assert m.total_return == 0.0 and m.n_trades == 0 and m.final_equity == 100_000.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backtest/test_metrics.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the metrics module**

Create `src/moneybot/backtest/metrics.py`:

```python
"""Pure performance math: trade log from Fills, plus headline metrics."""

from __future__ import annotations

from collections import deque
from datetime import date
from typing import TYPE_CHECKING

from moneybot.backtest.models import EquityPoint, PerformanceMetrics, TradeRecord

if TYPE_CHECKING:
    from moneybot.execution.models import Fill

_TRADING_DAYS = 252


def max_drawdown(equities: list[float]) -> float:
    """Worst peak-to-trough decline as a positive fraction (0.25 == -25%)."""
    peak = float("-inf")
    worst = 0.0
    for e in equities:
        peak = max(peak, e)
        if peak > 0:
            worst = min(worst, (e - peak) / peak)
    return abs(worst)


def _daily_returns(equities: list[float]) -> list[float]:
    out = []
    for prev, cur in zip(equities, equities[1:]):
        if prev != 0:
            out.append((cur - prev) / prev)
    return out


def sharpe(returns: list[float]) -> float:
    """Annualized Sharpe (risk-free = 0). Zero if <2 points or no variance."""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    if var <= 0:
        return 0.0
    std = var ** 0.5
    return (mean / std) * (_TRADING_DAYS ** 0.5)


def build_trade_log(fills: list[Fill]) -> list[TradeRecord]:
    """FIFO-match buys to sells per ticker; one TradeRecord per closed lot.

    Long-only (phase 1): 'buy' opens, 'sell' closes; rejected fills and shorts/covers
    are ignored. Open lots with no matching sell at the end are not reported."""
    open_lots: dict[str, deque] = {}
    trades: list[TradeRecord] = []
    for f in fills:
        if f.status != "filled" or f.filled_qty <= 0:
            continue
        if f.side == "buy":
            open_lots.setdefault(f.ticker, deque()).append(
                {"qty": f.filled_qty, "price": f.avg_price, "date": f.ts.date()}
            )
        elif f.side == "sell":
            remaining = f.filled_qty
            lots = open_lots.get(f.ticker)
            while remaining > 0 and lots:
                lot = lots[0]
                matched = min(remaining, lot["qty"])
                pnl = (f.avg_price - lot["price"]) * matched
                pnl_pct = (f.avg_price - lot["price"]) / lot["price"] if lot["price"] else 0.0
                trades.append(
                    TradeRecord(
                        ticker=f.ticker, qty=matched, entry_date=lot["date"], exit_date=f.ts.date(),
                        entry_price=lot["price"], exit_price=f.avg_price, pnl=pnl, pnl_pct=pnl_pct,
                        exit_reason=f.reason,
                    )
                )
                lot["qty"] -= matched
                remaining -= matched
                if lot["qty"] == 0:
                    lots.popleft()
    return trades


def compute_metrics(
    *,
    equity_curve: list[EquityPoint],
    trades: list[TradeRecord],
    starting_cash: float,
    benchmark_closes: list[float],
) -> PerformanceMetrics:
    equities = [p.equity for p in equity_curve]
    final_equity = equities[-1] if equities else starting_cash
    total_return = (final_equity - starting_cash) / starting_cash if starting_cash else 0.0

    returns = _daily_returns(equities)
    n_periods = len(returns)
    if n_periods > 0 and starting_cash > 0 and final_equity > 0:
        cagr = (final_equity / starting_cash) ** (_TRADING_DAYS / n_periods) - 1
    else:
        cagr = 0.0

    n_trades = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    win_rate = wins / n_trades if n_trades else 0.0

    if benchmark_closes:
        b0, bn = benchmark_closes[0], benchmark_closes[-1]
        benchmark_return = (bn - b0) / b0 if b0 else 0.0
        benchmark_final_equity = starting_cash * (1 + benchmark_return)
    else:
        benchmark_return = 0.0
        benchmark_final_equity = starting_cash

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        max_drawdown=max_drawdown(equities),
        sharpe=sharpe(returns),
        win_rate=win_rate,
        n_trades=n_trades,
        final_equity=final_equity,
        benchmark_return=benchmark_return,
        benchmark_final_equity=benchmark_final_equity,
    )
```

Remove the unused `date` import if ruff flags F401 (it is referenced only in type positions inside TradeRecord construction via `f.ts.date()`, which does not need the import — drop `from datetime import date` if unused).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backtest/test_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Run ruff and commit**

```bash
uv run ruff check src/moneybot/backtest/metrics.py
git add src/moneybot/backtest/metrics.py tests/backtest/test_metrics.py
git commit -m "feat(backtest): pure performance metrics + FIFO trade log"
```

---

### Task 8: Backtest engine (the driver)

`run_backtest` builds the orchestrator with the `SimClock`, an always-open market gate, and the caching research/analyst wrappers, then iterates the trading calendar calling `run_cycle(as_of=day)`. After each cycle it marks the portfolio to market for the equity curve and accumulates all fills. It returns a fully populated `BacktestReport`.

**Files:**
- Create: `src/moneybot/backtest/engine.py`
- Test: `tests/backtest/test_engine.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_engine.py`. This test wires real production components (DataLayer, PaperBroker via the real orchestrator factory, real Risk Engine) but with a **stub price provider** and a **FakeLLM**, so nothing hits the network or an LLM. It drives a tiny 3-day universe and asserts the harness produces an equity curve and metrics.

```python
from datetime import date

import pandas as pd

from moneybot.backtest.engine import run_backtest
from moneybot.backtest.models import BacktestConfig
from moneybot.cache import Cache
from moneybot.config import Settings, TickerMeta, Universe
from moneybot.data_layer import DataLayer
from moneybot.memory.models import MemoryContext


class StubPrices:
    """Deterministic rising price for NVDA; flat for SMH. Honors as_of (point-in-time)."""

    def get_bars(self, ticker, timeframe, lookback, as_of=None):
        base = {"NVDA": 100.0, "SMH": 200.0}.get(ticker, 50.0)
        # a single bar dated at as_of (enough for marking; rising via day number)
        bump = as_of.day if as_of else 1
        price = base + (bump if ticker == "NVDA" else 0)
        ts = pd.to_datetime([f"{as_of} 00:00:00+00:00"])
        return pd.DataFrame(
            {"ts": ts, "open": [price], "high": [price], "low": [price], "close": [price], "volume": [1_000_000]}
        )


class StubFilings:
    def get_recent_filings(self, ticker, types=None, since=None, as_of=None):
        return []


class StubNews:
    def get_news(self, query, since=None, as_of=None):
        return []


class StubRetriever:
    def retrieve(self, tickers, sector):
        return MemoryContext()


class FakeLLM:
    """Never produces a confirmed plan -> no entries; exercises the full no-trade path."""

    def complete_json(self, *, model, system, user, schema):
        return {}


def _universe():
    return Universe(
        sector="semiconductors", benchmark="SMH",
        tickers=[TickerMeta(symbol="NVDA"), TickerMeta(symbol="AMD")],
    )


def _data_layer(tmp_path):
    return DataLayer(
        _universe(), StubPrices(), Cache(str(tmp_path / "cache")),
        filings_provider=StubFilings(), news_provider=StubNews(),
    )


def _settings(tmp_path):
    return Settings(data_dir=str(tmp_path / "run"), paper_starting_cash=100_000.0)


def test_run_backtest_produces_curve_and_metrics(tmp_path):
    # Benchmark bars define the trading calendar; supply 3 weekdays.
    bench = pd.DataFrame({
        "ts": pd.to_datetime(["2024-03-04 00:00:00+00:00", "2024-03-05 00:00:00+00:00", "2024-03-06 00:00:00+00:00"]),
        "close": [200.0, 200.0, 200.0],
    })

    report = run_backtest(
        settings=_settings(tmp_path),
        data_layer=_data_layer(tmp_path),
        llm=FakeLLM(),
        retriever=StubRetriever(),
        config=BacktestConfig(start=date(2024, 3, 4), end=date(2024, 3, 6)),
        cache_root=tmp_path / "bt_cache",
        benchmark_bars=bench,
    )

    assert len(report.equity_curve) == 3
    assert report.equity_curve[0].day == date(2024, 3, 4)
    # No confirmed plans -> no trades, equity stays at starting cash.
    assert report.metrics.n_trades == 0
    assert report.metrics.final_equity == 100_000.0
    assert report.metrics.benchmark_return == 0.0
    assert any("daily-loss breaker" in n.lower() for n in report.notes)


def test_run_backtest_no_trading_days_is_safe(tmp_path):
    empty = pd.DataFrame({"ts": [], "close": []})
    report = run_backtest(
        settings=_settings(tmp_path),
        data_layer=_data_layer(tmp_path),
        llm=FakeLLM(),
        retriever=StubRetriever(),
        config=BacktestConfig(start=date(2024, 3, 4), end=date(2024, 3, 6)),
        cache_root=tmp_path / "bt_cache",
        benchmark_bars=empty,
    )
    assert report.equity_curve == []
    assert report.metrics.final_equity == 100_000.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backtest/test_engine.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the engine**

Create `src/moneybot/backtest/engine.py`:

```python
"""Drive the live orchestrator across historical days and assemble a report.

run_backtest is the single entry point. It is given an already-built data layer,
LLM, and retriever (the composition root wires real-or-cached providers); it builds
the orchestrator with a SimClock + always-open market gate + caching research/analyst,
replays each trading day through run_cycle(as_of=day), marks the portfolio to market
for the equity curve, and computes metrics vs the benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from moneybot.analyst.factory import build_analyst_agent
from moneybot.backtest.agent_cache import CachingAnalyst, CachingResearch
from moneybot.backtest.calendar import trading_days_from_bars
from moneybot.backtest.clock import SimClock
from moneybot.backtest.metrics import build_trade_log, compute_metrics
from moneybot.backtest.models import BacktestReport, EquityPoint
from moneybot.orchestrator.factory import build_orchestrator
from moneybot.orchestrator.portfolio import build_portfolio_state
from moneybot.research.factory import build_research_agent

if TYPE_CHECKING:
    from moneybot.backtest.models import BacktestConfig
    from moneybot.config import Settings
    from moneybot.data_layer import DataLayer
    from moneybot.llm.client import LLMClient
    from moneybot.memory.retriever import MemoryRetriever

_DAILY_BREAKER_NOTE = (
    "Daily-loss breaker is inert in a daily backtest: one cycle per day means "
    "start-of-day equity equals current equity, so day P&L is ~0. The breaker is "
    "an intraday protection; daily cadence cannot exercise it."
)


def run_backtest(
    *,
    settings: Settings,
    data_layer: DataLayer,
    llm: LLMClient,
    retriever: MemoryRetriever,
    config: BacktestConfig,
    cache_root: str | Path,
    benchmark_bars: pd.DataFrame,
) -> BacktestReport:
    cache_root = Path(cache_root)
    days = trading_days_from_bars(benchmark_bars, config.start, config.end)

    clock = SimClock()

    # Build (and cache-wrap) the AI layer; build_orchestrator uses these verbatim.
    if config.use_agents:
        research = CachingResearch(
            build_research_agent(settings=settings, data_layer=data_layer, retriever=retriever, llm=llm),
            root=cache_root, mode=config.mode,
        )
        analyst = CachingAnalyst(
            build_analyst_agent(settings=settings, data_layer=data_layer, retriever=retriever, llm=llm),
            root=cache_root, mode=config.mode,
        )
    else:
        research = _NoResearch()
        analyst = _NoAnalyst()

    orch = build_orchestrator(
        settings=settings,
        data_layer=data_layer,
        retriever=retriever,
        llm=llm,
        clock=clock,
        market_open=lambda _now: True,  # calendar already restricts to real trading days
        research=research,
        analyst=analyst,
    )

    equity_curve: list[EquityPoint] = []
    fills = []
    for day in days:
        clock.set_day(day)
        result = orch.run_cycle(as_of=day)
        fills.extend(result.entry_fills)
        fills.extend(result.exit_fills)
        point = _mark_equity(orch=orch, settings=settings, data_layer=data_layer, day=day)
        equity_curve.append(point)

    benchmark_closes = _benchmark_closes(benchmark_bars, days)
    trades = build_trade_log(fills)
    metrics = compute_metrics(
        equity_curve=equity_curve,
        trades=trades,
        starting_cash=config.starting_cash,
        benchmark_closes=benchmark_closes,
    )
    return BacktestReport(
        config=config,
        equity_curve=equity_curve,
        trades=trades,
        metrics=metrics,
        notes=[_DAILY_BREAKER_NOTE],
    )


def _mark_equity(*, orch, settings, data_layer, day) -> EquityPoint:
    """Marked-to-market equity for the day (reuses the live point-in-time marker)."""
    broker = orch.execution.broker
    try:
        state = build_portfolio_state(
            broker=broker, data_layer=data_layer, settings=settings, as_of=day, day_pnl_pct=0.0
        )
        equity = state.equity
    except ValueError:
        # Non-positive equity (e.g. blown account): fall back to broker cash.
        equity = broker.get_account().cash
    account = broker.get_account()
    positions = [p for p in broker.get_positions() if p.qty != 0]
    return EquityPoint(day=day, equity=equity, cash=account.cash, n_positions=len(positions))


def _benchmark_closes(benchmark_bars: pd.DataFrame, days: list) -> list[float]:
    if benchmark_bars.empty or not days:
        return []
    df = benchmark_bars.copy()
    df["_day"] = pd.to_datetime(df["ts"]).dt.date
    by_day = dict(zip(df["_day"], df["close"]))
    return [by_day[d] for d in days if d in by_day]


class _NoResearch:
    def research_universe(self, as_of=None):
        return {}


class _NoAnalyst:
    def analyze(self, research, as_of=None):
        return []
```

Note on the equity-curve `n_positions`: `PaperBroker.get_positions()` already excludes flats, but the `!= 0` filter is defensive against any provider that returns flats. Self-check ruff line length ≤ 100; the two long factory-call lines may need wrapping.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backtest/test_engine.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Run ruff and commit**

```bash
uv run ruff check src/moneybot/backtest/engine.py
git add src/moneybot/backtest/engine.py tests/backtest/test_engine.py
git commit -m "feat(backtest): engine driving run_cycle across historical days"
```

---

### Task 9: Report rendering + export

Turn a `BacktestReport` into a human-readable markdown summary (for the terminal) and machine-readable CSV/JSON files (equity curve + trade log) for further analysis.

**Files:**
- Create: `src/moneybot/backtest/report.py`
- Test: `tests/backtest/test_report.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_report.py`:

```python
from datetime import date

from moneybot.backtest.models import (
    BacktestConfig,
    BacktestReport,
    EquityPoint,
    PerformanceMetrics,
    TradeRecord,
)
from moneybot.backtest.report import render_summary, write_artifacts


def _report():
    return BacktestReport(
        config=BacktestConfig(start=date(2024, 1, 2), end=date(2024, 1, 31)),
        equity_curve=[
            EquityPoint(day=date(2024, 1, 2), equity=100_000.0, cash=100_000.0, n_positions=0),
            EquityPoint(day=date(2024, 1, 31), equity=112_000.0, cash=5_000.0, n_positions=2),
        ],
        trades=[
            TradeRecord(
                ticker="NVDA", qty=10, entry_date=date(2024, 1, 2), exit_date=date(2024, 1, 20),
                entry_price=100.0, exit_price=120.0, pnl=200.0, pnl_pct=0.20, exit_reason="profit_target",
            )
        ],
        metrics=PerformanceMetrics(
            total_return=0.12, cagr=2.0, max_drawdown=0.04, sharpe=1.5,
            win_rate=1.0, n_trades=1, final_equity=112_000.0,
            benchmark_return=0.05, benchmark_final_equity=105_000.0,
        ),
        notes=["daily-loss breaker note"],
    )


def test_render_summary_mentions_headline_numbers_and_benchmark():
    text = render_summary(_report())
    assert "Total return" in text
    assert "12.00%" in text          # strategy return formatted as percent
    assert "5.00%" in text           # benchmark return
    assert "SMH" in text or "benchmark" in text.lower()
    assert "Max drawdown" in text
    assert "daily-loss breaker note" in text


def test_write_artifacts_creates_files(tmp_path):
    paths = write_artifacts(_report(), out_dir=tmp_path)
    assert paths["equity_csv"].exists()
    assert paths["trades_csv"].exists()
    assert paths["report_json"].exists()
    equity_text = paths["equity_csv"].read_text(encoding="utf-8")
    assert "day,equity,cash,n_positions" in equity_text
    assert "2024-01-31" in equity_text
    trades_text = paths["trades_csv"].read_text(encoding="utf-8")
    assert "NVDA" in trades_text
    # JSON round-trips back into the model
    BacktestReport.model_validate_json(paths["report_json"].read_text(encoding="utf-8"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backtest/test_report.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the report module**

Create `src/moneybot/backtest/report.py`:

```python
"""Render a BacktestReport to a terminal summary and CSV/JSON artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from moneybot.backtest.models import BacktestReport


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def render_summary(report: BacktestReport) -> str:
    m = report.metrics
    cfg = report.config
    beat = m.total_return - m.benchmark_return
    lines = [
        "=== Backtest Report ===",
        f"Period:           {cfg.start} -> {cfg.end}  ({len(report.equity_curve)} trading days)",
        f"Starting cash:    ${cfg.starting_cash:,.0f}",
        "",
        f"Total return:     {_pct(m.total_return)}   (final ${m.final_equity:,.0f})",
        f"CAGR:             {_pct(m.cagr)}",
        f"Max drawdown:     {_pct(m.max_drawdown)}",
        f"Sharpe:           {m.sharpe:.2f}",
        f"Win rate:         {_pct(m.win_rate)}   ({m.n_trades} trades)",
        "",
        f"Buy & hold SMH:   {_pct(m.benchmark_return)}   (final ${m.benchmark_final_equity:,.0f})",
        f"Strategy vs SMH:  {_pct(beat)}   ({'beat' if beat >= 0 else 'trailed'} the benchmark)",
    ]
    if report.notes:
        lines.append("")
        lines.append("Notes:")
        lines.extend(f"  - {n}" for n in report.notes)
    return "\n".join(lines)


def write_artifacts(report: BacktestReport, *, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    equity_csv = out / "equity_curve.csv"
    trades_csv = out / "trades.csv"
    report_json = out / "report.json"

    with equity_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["day", "equity", "cash", "n_positions"])
        for p in report.equity_curve:
            w.writerow([p.day.isoformat(), p.equity, p.cash, p.n_positions])

    with trades_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ticker", "qty", "entry_date", "exit_date", "entry_price", "exit_price", "pnl", "pnl_pct", "exit_reason"])
        for t in report.trades:
            w.writerow([
                t.ticker, t.qty, t.entry_date.isoformat(), t.exit_date.isoformat(),
                t.entry_price, t.exit_price, t.pnl, t.pnl_pct, t.exit_reason,
            ])

    report_json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return {"equity_csv": equity_csv, "trades_csv": trades_csv, "report_json": report_json}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backtest/test_report.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/backtest/report.py tests/backtest/test_report.py
git commit -m "feat(backtest): terminal summary + CSV/JSON artifact export"
```

---

### Task 10: Package exports + CLI composition root

Export the public surface and add a `python -m moneybot.backtest` entry point that wires the *real* providers (with caches) from settings + `universe.yaml`, runs the backtest, prints the summary, and writes artifacts. The CLI is the composition root — the one place that constructs network-touching providers — so it has no network-free unit test; instead a small test imports it and exercises argument parsing with everything stubbed.

**Files:**
- Modify: `src/moneybot/backtest/__init__.py`
- Create: `src/moneybot/backtest/__main__.py`
- Test: `tests/backtest/test_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/backtest/test_cli.py`:

```python
from datetime import date

from moneybot.backtest import BacktestConfig, run_backtest  # public exports
from moneybot.backtest.__main__ import parse_args


def test_public_exports_exist():
    assert run_backtest is not None
    assert BacktestConfig is not None


def test_parse_args_minimal():
    ns = parse_args(["--start", "2024-01-01", "--end", "2024-06-30"])
    assert ns.start == date(2024, 1, 1)
    assert ns.end == date(2024, 6, 30)
    assert ns.mode == "record"        # default
    assert ns.timeframe == "1d"       # default


def test_parse_args_replay_and_cash():
    ns = parse_args(["--start", "2024-01-01", "--end", "2024-06-30", "--mode", "replay", "--cash", "50000"])
    assert ns.mode == "replay"
    assert ns.cash == 50000.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backtest/test_cli.py -v`
Expected: FAIL — cannot import `parse_args` / exports missing.

- [ ] **Step 3: Write exports + CLI**

Replace `src/moneybot/backtest/__init__.py` contents with:

```python
"""Backtesting harness: replay historical data through the live orchestrator code path."""

from moneybot.backtest.engine import run_backtest
from moneybot.backtest.models import (
    BacktestConfig,
    BacktestReport,
    EquityPoint,
    PerformanceMetrics,
    TradeRecord,
)
from moneybot.backtest.report import render_summary, write_artifacts

__all__ = [
    "run_backtest",
    "render_summary",
    "write_artifacts",
    "BacktestConfig",
    "BacktestReport",
    "EquityPoint",
    "PerformanceMetrics",
    "TradeRecord",
]
```

Create `src/moneybot/backtest/__main__.py`. The provider construction must mirror however the rest of the app builds the real data layer — **before writing this, read `src/moneybot/providers/` and any existing data-layer factory or composition code (e.g. a `build_data_layer`, the live entry point, or `__main__` elsewhere) and reuse the same provider classes and constructor arguments.** The skeleton below shows the structure; fill in the real provider construction to match the codebase (do not invent constructor signatures).

```python
"""CLI: python -m moneybot.backtest --start YYYY-MM-DD --end YYYY-MM-DD [--mode record|replay]

Composition root for backtests — the one place that constructs network-touching
providers. A 'record' run pays the LLM + data cost once and populates the cache;
'replay' runs are offline and free (reuse for Risk Engine parameter sweeps).

NOTE: the agent-output cache is keyed by date and assumes the research/analyst
configuration (universe, analyst_shortlist, rs_lookback_days, strategy ranking) is
unchanged. Tuning Risk Engine / exit parameters is safe against the cache; changing
research/analyst config requires a fresh --mode record run (or deleting the cache dir).
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from moneybot.backtest.engine import run_backtest
from moneybot.backtest.models import BacktestConfig
from moneybot.backtest.price_cache import CachingPriceProvider
from moneybot.backtest.report import render_summary, write_artifacts
from moneybot.cache import Cache
from moneybot.config import Settings, load_universe
from moneybot.data_layer import DataLayer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="moneybot.backtest", description="Replay history through the live bot.")
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--mode", choices=["record", "replay"], default="record")
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--cash", type=float, default=100_000.0)
    p.add_argument("--universe", default="universe.yaml")
    p.add_argument("--cache-dir", default="cache/backtest")
    p.add_argument("--out-dir", default="backtest_out")
    p.add_argument("--no-agents", action="store_true", help="mechanical-only: skip the AI layer")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    ns = parse_args(argv)
    settings = Settings()
    universe = load_universe(ns.universe)
    cache_root = Path(ns.cache_dir)

    # --- Construct the REAL providers here, matching the rest of the app. ---
    # Read src/moneybot/providers/ + any existing data-layer factory and reuse them.
    # Wrap the price provider so record runs populate the cache and replay runs are offline.
    real_price_provider = ...  # e.g. YFinancePriceProvider(...) per the codebase
    price_provider = CachingPriceProvider(real_price_provider, root=cache_root, mode=ns.mode)
    filings_provider = ...     # e.g. the real EDGAR filings provider
    news_provider = ...        # e.g. the real news provider
    data_layer = DataLayer(
        universe, price_provider, Cache(settings.cache_dir),
        filings_provider=filings_provider, news_provider=news_provider,
    )

    retriever = ...            # the real MemoryRetriever (or an empty-context stub for v1)
    llm = None                 # build_research/analyst lazily construct the real client when None

    # Trading calendar comes from the benchmark's real (cached on record) bars.
    benchmark_bars = data_layer.get_bars(
        universe.benchmark, ns.timeframe, _lookback_days(ns.start, ns.end), as_of=ns.end
    )

    config = BacktestConfig(
        start=ns.start, end=ns.end, timeframe=ns.timeframe,
        starting_cash=ns.cash, mode=ns.mode, use_agents=not ns.no_agents,
    )
    report = run_backtest(
        settings=settings, data_layer=data_layer, llm=llm, retriever=retriever,
        config=config, cache_root=cache_root, benchmark_bars=benchmark_bars,
    )
    print(render_summary(report))
    paths = write_artifacts(report, out_dir=ns.out_dir)
    print("\nArtifacts:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


def _lookback_days(start: date, end: date) -> int:
    # Enough calendar days to cover the range plus the risk lookback warmup.
    return (end - start).days + 60


if __name__ == "__main__":
    main()
```

The `...` placeholders for provider construction are intentional **scaffolding to be filled with the real provider classes from the codebase** — the implementer must read `src/moneybot/providers/` (and any existing live composition root) and substitute the actual constructors. The `parse_args` function and the wiring shape are complete and tested; `main()` itself is exercised by the operator, not the test suite.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backtest/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Smoke-check the module imports cleanly and commit**

Run: `uv run python -c "import moneybot.backtest, moneybot.backtest.__main__; print('ok')"`
Expected: prints `ok` (import-time errors would surface here).

```bash
uv run ruff check src/moneybot/backtest/
git add src/moneybot/backtest/__init__.py src/moneybot/backtest/__main__.py tests/backtest/test_cli.py
git commit -m "feat(backtest): public exports + CLI composition root"
```

---

### Task 11: Documentation

Add a plain-language "Phase 10: Backtesting" section to the README describing what the harness does, the record/replay workflow, the daily-cadence + daily-breaker limitations, and how the operator runs it to tune Risk Engine parameters.

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read the existing README phase sections**

Run: read `README.md` and locate the Phase 8 / Phase 9 prose bullets to match tone and format.

- [ ] **Step 2: Add the Phase 10 section**

Append a section in the same style as Phases 8–9. It must cover, in plain language:

```markdown
### Phase 10 — Backtesting harness

The backtester replays historical market data through the *exact same* code the bot
runs live — research → analyst → risk → execution → exits — one simulated trading day
at a time, with point-in-time data so it can never peek at the future. It's how we
check whether the strategy actually has an edge before risking real money.

- **Record once, replay free.** The first run ("record" mode) pays for the AI work
  (the Claude research + analyst calls) and the data downloads, and caches the AI's
  per-day decisions and the prices to disk. Every later run ("replay" mode) reuses
  that cache — no Claude calls, fully offline — so you can sweep Risk Engine settings
  (position size, stop-loss, profit target, exposure caps) cheaply and instantly.
  The cache is keyed by date and assumes the research/analyst setup is unchanged;
  if you change the universe or analyst settings, re-run in record mode.
- **Daily cadence.** Free price history is deep at daily resolution but only weeks
  deep intraday, so the backtest runs one cycle per trading day (trading days come
  from the sector ETF's real bar dates, so holidays are handled automatically).
- **What you get.** An equity curve (marked to market each day), total return, max
  drawdown, Sharpe, win rate, trade count, and a side-by-side comparison against just
  buying and holding the sector ETF (SMH) — printed as a summary and written as CSV +
  JSON for deeper analysis.
- **One limitation to know:** the intraday daily-loss circuit breaker can't be
  exercised by a daily backtest (with one cycle per day there's no intraday move to
  trip it). It still protects live/paper trading; the backtest just can't test it.

Run it:

​```bash
# First time (pays LLM + download cost, populates the cache):
uv run python -m moneybot.backtest --start 2024-01-01 --end 2024-12-31 --mode record

# Re-run after changing Risk Engine settings (free, offline):
uv run python -m moneybot.backtest --start 2024-01-01 --end 2024-12-31 --mode replay
​```
```

(Remove the zero-width characters around the inner code fence; they are only here to keep this plan's markdown from terminating early. Write a normal nested ```bash block.)

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add Phase 10 backtesting harness section to README"
```

---

### Final Verification (after all tasks)

- [ ] **Run the full suite:** `uv run pytest -q` — expect all prior tests plus the new backtest tests green (Plan 9 left 321 passing; this plan adds roughly 30+).
- [ ] **Lint:** `uv run ruff check src/moneybot/backtest src/moneybot/orchestrator/factory.py` — clean. Self-check line length ≤ 100 (not enforced by the default rule set).
- [ ] **Import smoke:** `uv run python -c "import moneybot.backtest; print('ok')"`.
- [ ] **No-network / no-LLM discipline:** confirm no test under `tests/backtest/` imports a real provider, the `anthropic` SDK, or makes a network call; all use stubs/fakes + `tmp_path`.
- [ ] **No fabricated clock:** confirm no backtest *component* calls `datetime.now`/`date.today`; the only real-clock default remains in `build_orchestrator` (and is overridden by `SimClock` here).
- [ ] **Dispatch the final whole-implementation code review** (Opus) covering: point-in-time correctness (as_of threaded everywhere, no lookahead), cache determinism + record/replay safety, marked-to-market equity correctness, FIFO trade-log math, and that the harness drives the *real* orchestrator code path (no shadow re-implementation of the cycle).

---

## Self-Review (author checklist — completed)

- **Spec coverage:** §4.8 "Backtesting harness — replays historical data through the same Analyst/Risk/Execution code path (point-in-time, no lookahead); also used to tune Risk Engine parameters" → Tasks 1–10 drive the real `run_cycle`; point-in-time via `as_of` + `SimClock`; tuning enabled by record/replay caching. Build-sequence step 9 ("Backtesting harness over the same code path") → satisfied. Observability/daily-summary remains deferred (Phase 11), consistent with the Plan 9 scope decision.
- **Placeholder scan:** the only `...` are in `__main__.py` provider construction, explicitly flagged as composition-root wiring the implementer fills from the codebase (a real value cannot be hard-coded without inventing provider constructors). Every test and library module is complete code.
- **Type consistency:** `BacktestConfig`/`EquityPoint`/`TradeRecord`/`PerformanceMetrics`/`BacktestReport` field names are identical across Tasks 2, 7, 8, 9, 10. `run_backtest` signature in Task 8 matches its calls in Tasks 8-test and 10. `CachingResearch.research_universe` / `CachingAnalyst.analyze` match the methods `Orchestrator.run_cycle` calls. `build_orchestrator(research=, analyst=)` added in Task 1 is consumed in Task 8.
```
