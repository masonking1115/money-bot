# Execution Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the layer that turns the Risk Engine's approved `RiskDecision`s (and optional `HedgeOrder`) into real broker orders, tracks fills, persists the bot's believed positions, and reconciles them against the broker — paper or live by a single config flag.

**Architecture:** A `Broker` Protocol seam (mirroring the `LLMClient` / `PriceProvider` seams) has two implementations: `PaperBroker` (in-memory simulator, the validated default path) and `AlpacaBroker` (thin live adapter whose SDK calls are isolated behind patchable `_*_raw` methods so no test hits the network). A pure `apply_fill` cost-basis function is shared by the paper broker and the JSON-persisted `PositionStore`. The `ExecutionAdapter` orchestrates: it translates a `RiskAssessment` into idempotent orders (keyed by a deterministic `client_order_id`), places them through whichever broker is wired, updates the store on filled orders, and offers report-only reconciliation. A future IBKR broker is just one more `Broker` implementation — nothing else changes.

**Tech Stack:** Python 3.12, Pydantic v2, `alpaca-py` (live only, lazily imported), pytest, uv, ruff. Mirrors existing moneybot conventions: `from __future__ import annotations` in production modules (omitted in tests), type-only imports under `TYPE_CHECKING`, injected clock (no fabricated `datetime.now`), JSON persistence under a root dir, `build_*(*, settings, ...)` factories that lazily construct the real adapter only when no override is passed. Ruff line-length convention is 100 (self-check — the default rule set does not flag E501). NO test may hit the network.

---

## Conventions every task must follow

- **Production modules** start with `from __future__ import annotations`. **Test files do NOT** (matches the existing suite).
- **Type-only imports** (`Settings`, `RiskAssessment`, `datetime` used only in signatures that are stringized) go under `if TYPE_CHECKING:`. Runtime-needed imports (`datetime`, `timezone` for the default clock; pydantic types) are normal imports.
- **No fabricated clock.** Timestamps come from an injected `clock: Callable[[], datetime] | None`, defaulting to `lambda: datetime.now(timezone.utc)` — identical to `JournalStore` in `src/moneybot/memory/journal.py`.
- **Run tests with** `uv run pytest <path> -v` and **lint with** `uv run ruff check src/moneybot/execution tests/execution`.
- **Sign convention** (consistent with `src/moneybot/risk/models.py` `Position`): long quantities are **positive**, short quantities are **negative**.
- **Idempotency:** every order carries a deterministic `client_order_id`. Placing the same `client_order_id` twice must not double-apply — enforced in both `PaperBroker` and `PositionStore`. This is a real-money safety property (a re-run cycle must not double-trade or double-count).

---

## File Structure

- Create: `src/moneybot/execution/__init__.py` — package marker + public exports
- Create: `src/moneybot/execution/models.py` — typed order/fill/position/reconcile models
- Create: `src/moneybot/execution/positions.py` — pure `apply_fill` cost-basis math
- Create: `src/moneybot/execution/broker.py` — `Broker` Protocol seam
- Create: `src/moneybot/execution/paper.py` — `PaperBroker` simulator
- Create: `src/moneybot/execution/store.py` — `PositionStore` (JSON + applied-fill ledger)
- Create: `src/moneybot/execution/reconcile.py` — pure `reconcile` drift report
- Create: `src/moneybot/execution/adapter.py` — `ExecutionAdapter`
- Create: `src/moneybot/execution/alpaca.py` — `AlpacaBroker` thin live adapter
- Create: `src/moneybot/execution/factory.py` — `build_execution_adapter`
- Modify: `src/moneybot/config.py` — add `paper_starting_cash` setting
- Modify: `README.md` — add Phase 8 bullet
- Test: `tests/execution/__init__.py`, and one test module per source file below

---

### Task 1: Config — paper starting cash

**Files:**
- Modify: `src/moneybot/config.py` (after the `# Risk Engine` block, lines ~42-50)
- Test: `tests/execution/__init__.py` (Create, empty), `tests/execution/test_config_execution.py` (Create)

- [ ] **Step 1: Create the test package marker**

Create `tests/execution/__init__.py` as an empty file (matches `tests/risk/__init__.py`).

- [ ] **Step 2: Write the failing test**

Create `tests/execution/test_config_execution.py`:

```python
from moneybot.config import Settings


def test_paper_starting_cash_default():
    s = Settings()
    assert s.paper_starting_cash == 100_000.0


def test_paper_starting_cash_overridable(monkeypatch):
    monkeypatch.setenv("MONEYBOT_PAPER_STARTING_CASH", "250000")
    s = Settings()
    assert s.paper_starting_cash == 250_000.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/execution/test_config_execution.py -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'paper_starting_cash'`

- [ ] **Step 4: Add the setting**

In `src/moneybot/config.py`, immediately after the `kill_switch_file` line inside `Settings`, add:

```python

    # Execution
    paper_starting_cash: float = 100_000.0  # simulated account equity for the paper broker
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/execution/test_config_execution.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add src/moneybot/config.py tests/execution/__init__.py tests/execution/test_config_execution.py
git commit -m "feat(execution): add paper_starting_cash setting"
```

---

### Task 2: Models

**Files:**
- Create: `src/moneybot/execution/__init__.py`
- Create: `src/moneybot/execution/models.py`
- Test: `tests/execution/test_models.py`

- [ ] **Step 1: Create the package marker**

Create `src/moneybot/execution/__init__.py`:

```python
"""Execution Adapter: places approved orders (paper or live), tracks fills,
persists positions, and reconciles against the broker. No LLM."""
```

(Public exports are added in Task 11 once the classes exist — leaving them out now keeps the package importable while later tasks build the modules.)

- [ ] **Step 2: Write the failing test**

Create `tests/execution/test_models.py`:

```python
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from moneybot.execution.models import (
    AccountSnapshot,
    Discrepancy,
    Fill,
    OrderRequest,
    PositionRecord,
    ReconciliationResult,
)


def test_order_request_defaults_to_market():
    o = OrderRequest(
        client_order_id="c1:NVDA:buy", ticker="NVDA", side="buy", quantity=10
    )
    assert o.order_type == "market"
    assert o.reference_price is None


def test_order_request_quantity_must_be_positive():
    with pytest.raises(ValidationError):
        OrderRequest(client_order_id="c1", ticker="NVDA", side="buy", quantity=0)


def test_order_request_rejects_unknown_side():
    with pytest.raises(ValidationError):
        OrderRequest(client_order_id="c1", ticker="NVDA", side="hodl", quantity=1)


def test_fill_construction():
    f = Fill(
        client_order_id="c1:NVDA:buy",
        broker_order_id="paper-1",
        ticker="NVDA",
        side="buy",
        status="filled",
        filled_qty=10,
        avg_price=100.0,
        ts=datetime(2026, 6, 9, tzinfo=timezone.utc),
    )
    assert f.status == "filled"
    assert f.reason == ""


def test_position_record_signed_qty():
    long = PositionRecord(ticker="NVDA", qty=10.0, avg_price=100.0)
    short = PositionRecord(ticker="SMH", qty=-5.0, avg_price=200.0)
    assert long.qty > 0 and short.qty < 0


def test_account_snapshot():
    a = AccountSnapshot(equity=100_000.0, cash=40_000.0)
    assert a.equity == 100_000.0


def test_reconciliation_result_in_sync():
    r = ReconciliationResult(in_sync=True, discrepancies=[])
    assert r.in_sync and r.discrepancies == []


def test_discrepancy_fields():
    d = Discrepancy(ticker="NVDA", stored_qty=10.0, broker_qty=8.0)
    assert d.ticker == "NVDA"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/execution/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'moneybot.execution.models'`

- [ ] **Step 4: Implement the models**

Create `src/moneybot/execution/models.py`:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/execution/test_models.py -v`
Expected: PASS (8 passed)

- [ ] **Step 6: Commit**

```bash
git add src/moneybot/execution/__init__.py src/moneybot/execution/models.py tests/execution/test_models.py
git commit -m "feat(execution): order/fill/position/reconcile models"
```

---

### Task 3: Pure cost-basis math (`apply_fill`)

This pure function is the single source of truth for how a fill mutates a position. Both `PaperBroker` (its own book) and `PositionStore` (the bot's belief) call it — so the math lives in exactly one place.

**Files:**
- Create: `src/moneybot/execution/positions.py`
- Test: `tests/execution/test_positions.py`

- [ ] **Step 1: Write the failing test**

Create `tests/execution/test_positions.py`:

```python
from datetime import datetime, timezone

from moneybot.execution.models import Fill, PositionRecord
from moneybot.execution.positions import apply_fill


def _fill(side, qty, price, ticker="NVDA"):
    return Fill(
        client_order_id=f"c:{ticker}:{side}",
        broker_order_id="b1",
        ticker=ticker,
        side=side,
        status="filled",
        filled_qty=qty,
        avg_price=price,
        ts=datetime(2026, 6, 9, tzinfo=timezone.utc),
    )


def test_open_long_from_flat():
    r = apply_fill(None, _fill("buy", 10, 100.0))
    assert r.qty == 10.0 and r.avg_price == 100.0 and r.ticker == "NVDA"


def test_add_to_long_weighted_average():
    start = PositionRecord(ticker="NVDA", qty=10.0, avg_price=100.0)
    r = apply_fill(start, _fill("buy", 10, 120.0))
    assert r.qty == 20.0 and r.avg_price == 110.0  # (10*100 + 10*120)/20


def test_reduce_long_keeps_cost_basis():
    start = PositionRecord(ticker="NVDA", qty=10.0, avg_price=100.0)
    r = apply_fill(start, _fill("sell", 4, 130.0))
    assert r.qty == 6.0 and r.avg_price == 100.0  # cost basis unchanged on a partial exit


def test_fully_closing_returns_none():
    start = PositionRecord(ticker="NVDA", qty=10.0, avg_price=100.0)
    r = apply_fill(start, _fill("sell", 10, 130.0))
    assert r is None


def test_open_short_from_flat():
    r = apply_fill(None, _fill("short", 5, 200.0))
    assert r.qty == -5.0 and r.avg_price == 200.0


def test_cover_reduces_short():
    start = PositionRecord(ticker="SMH", qty=-5.0, avg_price=200.0)
    r = apply_fill(start, _fill("cover", 2, 180.0, ticker="SMH"))
    assert r.qty == -3.0 and r.avg_price == 200.0


def test_crossing_through_zero_resets_avg_to_fill_price():
    # long 10, sell 15 -> net short 5 at the fill price
    start = PositionRecord(ticker="NVDA", qty=10.0, avg_price=100.0)
    r = apply_fill(start, _fill("sell", 15, 130.0))
    assert r.qty == -5.0 and r.avg_price == 130.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/execution/test_positions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'moneybot.execution.positions'`

- [ ] **Step 3: Implement `apply_fill`**

Create `src/moneybot/execution/positions.py`:

```python
"""Pure cost-basis math: how one fill mutates a position.

Single source of truth for both the paper broker's book and the position store.
qty is signed (long positive, short negative). buy/cover move qty up; sell/short
move it down. Cost basis is a weighted average while adding in the same
direction, is left unchanged while reducing, and resets to the fill price when a
position crosses through zero. Returns None when the position goes flat.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from moneybot.execution.models import PositionRecord

if TYPE_CHECKING:
    from moneybot.execution.models import Fill

_FLAT = 1e-9


def apply_fill(current: PositionRecord | None, fill: Fill) -> PositionRecord | None:
    signed = float(fill.filled_qty)
    if fill.side in ("sell", "short"):
        signed = -signed

    old_qty = current.qty if current is not None else 0.0
    old_avg = current.avg_price if current is not None else 0.0
    new_qty = old_qty + signed

    if abs(new_qty) < _FLAT:
        return None  # position closed out

    same_direction = old_qty == 0.0 or (old_qty > 0) == (new_qty > 0)
    if not same_direction:
        new_avg = fill.avg_price  # crossed zero: remaining shares are at the fill price
    elif abs(new_qty) > abs(old_qty):
        # adding in the same direction -> weighted-average cost
        new_avg = (abs(old_qty) * old_avg + abs(signed) * fill.avg_price) / abs(new_qty)
    else:
        new_avg = old_avg  # reducing -> cost basis unchanged

    return PositionRecord(ticker=fill.ticker, qty=new_qty, avg_price=new_avg)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/execution/test_positions.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/execution/positions.py tests/execution/test_positions.py
git commit -m "feat(execution): pure apply_fill cost-basis math"
```

---

### Task 4: Broker Protocol seam

**Files:**
- Create: `src/moneybot/execution/broker.py`
- Test: `tests/execution/test_broker.py`

- [ ] **Step 1: Write the failing test**

Create `tests/execution/test_broker.py`:

```python
from datetime import datetime, timezone

from moneybot.execution.broker import Broker
from moneybot.execution.models import AccountSnapshot, Fill, OrderRequest, PositionRecord


class _StubBroker:
    def place_order(self, order: OrderRequest) -> Fill:
        return Fill(
            client_order_id=order.client_order_id,
            broker_order_id="x",
            ticker=order.ticker,
            side=order.side,
            status="filled",
            filled_qty=order.quantity,
            avg_price=1.0,
            ts=datetime(2026, 6, 9, tzinfo=timezone.utc),
        )

    def get_positions(self) -> list[PositionRecord]:
        return []

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(equity=1.0, cash=1.0)


def test_stub_satisfies_broker_protocol():
    assert isinstance(_StubBroker(), Broker)


def test_non_broker_fails_protocol_check():
    assert not isinstance(object(), Broker)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/execution/test_broker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'moneybot.execution.broker'`

- [ ] **Step 3: Implement the Protocol**

Create `src/moneybot/execution/broker.py`:

```python
"""The broker seam: the single contract the Execution Adapter depends on.

PaperBroker and AlpacaBroker implement it; a future IBKR broker is just one more
implementation, with no change to the adapter, store, or reconciliation. Mirrors
the LLMClient / PriceProvider Protocol seams elsewhere in moneybot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from moneybot.execution.models import (
        AccountSnapshot,
        Fill,
        OrderRequest,
        PositionRecord,
    )


@runtime_checkable
class Broker(Protocol):
    def place_order(self, order: OrderRequest) -> Fill:
        """Place one order and return the resulting Fill.

        Implementations must be idempotent on order.client_order_id: placing the
        same client_order_id twice must not result in two distinct trades.
        """
        ...

    def get_positions(self) -> list[PositionRecord]:
        """Return current non-flat positions (qty signed: long +, short -)."""
        ...

    def get_account(self) -> AccountSnapshot:
        """Return top-line account equity and cash."""
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/execution/test_broker.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/execution/broker.py tests/execution/test_broker.py
git commit -m "feat(execution): Broker Protocol seam"
```

---

### Task 5: PaperBroker

**Files:**
- Create: `src/moneybot/execution/paper.py`
- Test: `tests/execution/test_paper.py`

- [ ] **Step 1: Write the failing test**

Create `tests/execution/test_paper.py`:

```python
from datetime import datetime, timezone

from moneybot.execution.models import OrderRequest
from moneybot.execution.paper import PaperBroker


def _clock():
    return datetime(2026, 6, 9, tzinfo=timezone.utc)


def _broker(cash=100_000.0):
    return PaperBroker(starting_cash=cash, clock=_clock)


def _buy(ticker="NVDA", qty=10, price=100.0, oid="c1:NVDA:buy"):
    return OrderRequest(
        client_order_id=oid, ticker=ticker, side="buy", quantity=qty, reference_price=price
    )


def test_buy_fills_and_decrements_cash():
    b = _broker()
    fill = b.place_order(_buy())
    assert fill.status == "filled"
    assert fill.filled_qty == 10 and fill.avg_price == 100.0
    assert fill.broker_order_id == "paper-1"
    assert fill.ts == _clock()
    positions = b.get_positions()
    assert len(positions) == 1
    assert positions[0].ticker == "NVDA" and positions[0].qty == 10.0
    acct = b.get_account()
    assert acct.cash == 100_000.0 - 1_000.0
    assert acct.equity == 100_000.0  # mark-to-cost: cash + qty*avg_price


def test_missing_reference_price_is_rejected():
    b = _broker()
    order = OrderRequest(client_order_id="c2", ticker="NVDA", side="buy", quantity=5)
    fill = b.place_order(order)
    assert fill.status == "rejected"
    assert fill.filled_qty == 0
    assert "reference price" in fill.reason
    assert b.get_positions() == []


def test_idempotent_on_client_order_id():
    b = _broker()
    first = b.place_order(_buy())
    second = b.place_order(_buy())  # same client_order_id
    assert second.broker_order_id == first.broker_order_id
    assert b.get_account().cash == 100_000.0 - 1_000.0  # applied once only
    assert b.get_positions()[0].qty == 10.0


def test_short_adds_cash_and_makes_negative_position():
    b = _broker()
    order = OrderRequest(
        client_order_id="c3:SMH:short",
        ticker="SMH",
        side="short",
        quantity=5,
        reference_price=200.0,
    )
    fill = b.place_order(order)
    assert fill.status == "filled"
    assert b.get_account().cash == 100_000.0 + 1_000.0
    pos = b.get_positions()[0]
    assert pos.ticker == "SMH" and pos.qty == -5.0


def test_flat_positions_excluded():
    b = _broker()
    b.place_order(_buy(qty=10, oid="o-buy"))
    sell = OrderRequest(
        client_order_id="o-sell",
        ticker="NVDA",
        side="sell",
        quantity=10,
        reference_price=110.0,
    )
    b.place_order(sell)
    assert b.get_positions() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/execution/test_paper.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'moneybot.execution.paper'`

- [ ] **Step 3: Implement PaperBroker**

Create `src/moneybot/execution/paper.py`:

```python
"""PaperBroker: a deterministic in-memory simulated broker (no network, no LLM).

Fills market orders instantly at the order's reference_price (rejecting if none
is supplied). Tracks cash and positions internally. Equity is marked-to-cost
(cash + sum(qty * avg_price)) — the orchestrator re-marks to market with live
prices when it builds a PortfolioState. Idempotent on client_order_id: a repeated
order returns the original fill without re-applying it. This is the validated
default path; the same code runs in backtests and paper trading.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from moneybot.execution.models import AccountSnapshot, Fill, PositionRecord
from moneybot.execution.positions import apply_fill

if TYPE_CHECKING:
    from moneybot.execution.models import OrderRequest


class PaperBroker:
    def __init__(
        self,
        *,
        starting_cash: float,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.cash = starting_cash
        self._positions: dict[str, PositionRecord] = {}
        self._fills: dict[str, Fill] = {}  # client_order_id -> Fill (idempotency)
        self._seq = 0
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def place_order(self, order: OrderRequest) -> Fill:
        prior = self._fills.get(order.client_order_id)
        if prior is not None:
            return prior  # idempotent: do not re-apply

        price = order.reference_price
        if price is None or price <= 0:
            return Fill(
                client_order_id=order.client_order_id,
                broker_order_id="paper-rejected",
                ticker=order.ticker,
                side=order.side,
                status="rejected",
                ts=self._clock(),
                reason="no reference price to fill against",
            )

        self._seq += 1
        fill = Fill(
            client_order_id=order.client_order_id,
            broker_order_id=f"paper-{self._seq}",
            ticker=order.ticker,
            side=order.side,
            status="filled",
            filled_qty=order.quantity,
            avg_price=price,
            ts=self._clock(),
        )

        # buy/cover cost cash; sell/short add cash.
        signed = order.quantity if order.side in ("buy", "cover") else -order.quantity
        self.cash -= signed * price

        updated = apply_fill(self._positions.get(order.ticker), fill)
        if updated is None:
            self._positions.pop(order.ticker, None)
        else:
            self._positions[order.ticker] = updated

        self._fills[order.client_order_id] = fill
        return fill

    def get_positions(self) -> list[PositionRecord]:
        return list(self._positions.values())

    def get_account(self) -> AccountSnapshot:
        marked = sum(p.qty * p.avg_price for p in self._positions.values())
        return AccountSnapshot(equity=self.cash + marked, cash=self.cash)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/execution/test_paper.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/execution/paper.py tests/execution/test_paper.py
git commit -m "feat(execution): PaperBroker simulator with idempotent fills"
```

---

### Task 6: PositionStore (JSON + applied-fill ledger)

**Files:**
- Create: `src/moneybot/execution/store.py`
- Test: `tests/execution/test_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/execution/test_store.py`:

```python
from datetime import datetime, timezone

from moneybot.execution.models import Fill
from moneybot.execution.store import PositionStore


def _fill(side, qty, price, ticker="NVDA", oid="c1"):
    return Fill(
        client_order_id=oid,
        broker_order_id="b1",
        ticker=ticker,
        side=side,
        status="filled",
        filled_qty=qty,
        avg_price=price,
        ts=datetime(2026, 6, 9, tzinfo=timezone.utc),
    )


def test_empty_store_returns_nothing(tmp_path):
    store = PositionStore(tmp_path)
    assert store.get_all() == []


def test_apply_fill_creates_position(tmp_path):
    store = PositionStore(tmp_path)
    store.apply_fill(_fill("buy", 10, 100.0, oid="o1"))
    positions = store.get_all()
    assert len(positions) == 1
    assert positions[0].ticker == "NVDA" and positions[0].qty == 10.0


def test_persists_across_instances(tmp_path):
    PositionStore(tmp_path).apply_fill(_fill("buy", 10, 100.0, oid="o1"))
    reopened = PositionStore(tmp_path)
    assert reopened.get_all()[0].qty == 10.0


def test_apply_fill_is_idempotent_on_client_order_id(tmp_path):
    store = PositionStore(tmp_path)
    store.apply_fill(_fill("buy", 10, 100.0, oid="dup"))
    store.apply_fill(_fill("buy", 10, 100.0, oid="dup"))  # same id -> no double count
    assert store.get_all()[0].qty == 10.0


def test_distinct_ids_accumulate(tmp_path):
    store = PositionStore(tmp_path)
    store.apply_fill(_fill("buy", 10, 100.0, oid="o1"))
    store.apply_fill(_fill("buy", 10, 120.0, oid="o2"))
    pos = store.get_all()[0]
    assert pos.qty == 20.0 and pos.avg_price == 110.0


def test_closing_removes_position(tmp_path):
    store = PositionStore(tmp_path)
    store.apply_fill(_fill("buy", 10, 100.0, oid="o1"))
    store.apply_fill(_fill("sell", 10, 130.0, oid="o2"))
    assert store.get_all() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/execution/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'moneybot.execution.store'`

- [ ] **Step 3: Implement PositionStore**

Create `src/moneybot/execution/store.py`:

```python
"""PositionStore: the bot's own belief about what it holds, persisted as JSON.

A single positions.json under the data root holds the current positions plus an
"applied" ledger of client_order_ids already folded in — so re-applying a fill
(e.g. a re-run cycle) is a no-op. This belief is what reconciliation compares
against the broker's reported truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from moneybot.execution.models import PositionRecord
from moneybot.execution.positions import apply_fill

if TYPE_CHECKING:
    from moneybot.execution.models import Fill


class PositionStore:
    def __init__(self, root: str | Path) -> None:
        self.path = Path(root) / "positions.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> tuple[dict[str, PositionRecord], set[str]]:
        if not self.path.exists():
            return {}, set()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        positions = {
            ticker: PositionRecord.model_validate(rec)
            for ticker, rec in data.get("positions", {}).items()
        }
        applied = set(data.get("applied", []))
        return positions, applied

    def _write(self, positions: dict[str, PositionRecord], applied: set[str]) -> None:
        data = {
            "positions": {t: p.model_dump() for t, p in positions.items()},
            "applied": sorted(applied),
        }
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_all(self) -> list[PositionRecord]:
        positions, _ = self._read()
        return list(positions.values())

    def apply_fill(self, fill: Fill) -> None:
        positions, applied = self._read()
        if fill.client_order_id in applied:
            return  # already folded in — idempotent
        updated = apply_fill(positions.get(fill.ticker), fill)
        if updated is None:
            positions.pop(fill.ticker, None)
        else:
            positions[fill.ticker] = updated
        applied.add(fill.client_order_id)
        self._write(positions, applied)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/execution/test_store.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/execution/store.py tests/execution/test_store.py
git commit -m "feat(execution): JSON PositionStore with idempotent fill ledger"
```

---

### Task 7: Reconciliation (pure)

**Files:**
- Create: `src/moneybot/execution/reconcile.py`
- Test: `tests/execution/test_reconcile.py`

- [ ] **Step 1: Write the failing test**

Create `tests/execution/test_reconcile.py`:

```python
from moneybot.execution.models import PositionRecord
from moneybot.execution.reconcile import reconcile


def _p(ticker, qty):
    return PositionRecord(ticker=ticker, qty=qty, avg_price=100.0)


def test_in_sync_when_quantities_match():
    stored = [_p("NVDA", 10.0), _p("AMD", 5.0)]
    broker = [_p("AMD", 5.0), _p("NVDA", 10.0)]  # order-independent
    result = reconcile(stored, broker)
    assert result.in_sync
    assert result.discrepancies == []


def test_quantity_mismatch_is_a_discrepancy():
    result = reconcile([_p("NVDA", 10.0)], [_p("NVDA", 8.0)])
    assert not result.in_sync
    assert len(result.discrepancies) == 1
    d = result.discrepancies[0]
    assert d.ticker == "NVDA" and d.stored_qty == 10.0 and d.broker_qty == 8.0


def test_position_missing_at_broker():
    result = reconcile([_p("NVDA", 10.0)], [])
    assert not result.in_sync
    assert result.discrepancies[0].broker_qty == 0.0


def test_unexpected_position_at_broker():
    result = reconcile([], [_p("NVDA", 4.0)])
    assert not result.in_sync
    assert result.discrepancies[0].stored_qty == 0.0


def test_tiny_float_drift_is_tolerated():
    result = reconcile([_p("NVDA", 10.0)], [_p("NVDA", 10.0 + 1e-9)])
    assert result.in_sync


def test_discrepancies_sorted_by_ticker():
    result = reconcile([_p("NVDA", 1.0), _p("AMD", 1.0)], [])
    assert [d.ticker for d in result.discrepancies] == ["AMD", "NVDA"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/execution/test_reconcile.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'moneybot.execution.reconcile'`

- [ ] **Step 3: Implement reconcile**

Create `src/moneybot/execution/reconcile.py`:

```python
"""Pure reconciliation: compare the bot's stored positions against the broker's.

Report-only. It never places orders to "fix" drift — surfacing a discrepancy is
a safety event for the operator/orchestrator to handle, not something to auto-
trade away. A position present on only one side reports the other side as 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from moneybot.execution.models import Discrepancy, ReconciliationResult

if TYPE_CHECKING:
    from moneybot.execution.models import PositionRecord

_TOLERANCE = 1e-6


def reconcile(
    stored: list[PositionRecord],
    broker: list[PositionRecord],
) -> ReconciliationResult:
    stored_qty = {p.ticker: p.qty for p in stored}
    broker_qty = {p.ticker: p.qty for p in broker}

    discrepancies: list[Discrepancy] = []
    for ticker in sorted(set(stored_qty) | set(broker_qty)):
        s = stored_qty.get(ticker, 0.0)
        b = broker_qty.get(ticker, 0.0)
        if abs(s - b) > _TOLERANCE:
            discrepancies.append(Discrepancy(ticker=ticker, stored_qty=s, broker_qty=b))

    return ReconciliationResult(in_sync=not discrepancies, discrepancies=discrepancies)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/execution/test_reconcile.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/execution/reconcile.py tests/execution/test_reconcile.py
git commit -m "feat(execution): pure report-only reconciliation"
```

---

### Task 8: ExecutionAdapter

**Files:**
- Create: `src/moneybot/execution/adapter.py`
- Test: `tests/execution/test_adapter.py`

- [ ] **Step 1: Write the failing test**

Create `tests/execution/test_adapter.py`:

```python
from datetime import datetime, timezone

from moneybot.execution.adapter import ExecutionAdapter
from moneybot.execution.models import AccountSnapshot, Fill, PositionRecord
from moneybot.execution.store import PositionStore
from moneybot.risk.models import HedgeOrder, RiskAssessment, RiskDecision


class FakeBroker:
    """Records placed orders; fills buys/shorts at their reference price."""

    def __init__(self):
        self.orders = []
        self._seq = 0

    def place_order(self, order):
        self.orders.append(order)
        self._seq += 1
        return Fill(
            client_order_id=order.client_order_id,
            broker_order_id=f"fake-{self._seq}",
            ticker=order.ticker,
            side=order.side,
            status="filled",
            filled_qty=order.quantity,
            avg_price=order.reference_price,
            ts=datetime(2026, 6, 9, tzinfo=timezone.utc),
        )

    def get_positions(self):
        return []

    def get_account(self):
        return AccountSnapshot(equity=100_000.0, cash=100_000.0)


def _approved(ticker="NVDA", shares=10, price=100.0):
    return RiskDecision(
        ticker=ticker,
        approved=True,
        target_weight=0.01,
        target_dollars=shares * price,
        shares=shares,
        reference_price=price,
        reasoning="approved within limits",
    )


def _vetoed(ticker="AMD"):
    return RiskDecision(
        ticker=ticker, approved=False, rules_fired=["liquidity"], reasoning="thin"
    )


def test_execute_places_approved_orders_and_updates_store(tmp_path):
    broker = FakeBroker()
    store = PositionStore(tmp_path)
    adapter = ExecutionAdapter(broker=broker, store=store)

    assessment = RiskAssessment(decisions=[_approved(), _vetoed()])
    fills = adapter.execute(assessment, cycle_id="cycle-1")

    assert len(broker.orders) == 1  # only the approved one
    assert broker.orders[0].client_order_id == "cycle-1:NVDA:buy"
    assert broker.orders[0].side == "buy"
    assert len(fills) == 1 and fills[0].status == "filled"
    assert store.get_all()[0].ticker == "NVDA" and store.get_all()[0].qty == 10.0


def test_halted_assessment_places_nothing(tmp_path):
    broker = FakeBroker()
    adapter = ExecutionAdapter(broker=broker, store=PositionStore(tmp_path))
    assessment = RiskAssessment(
        decisions=[_vetoed("NVDA")], halted=True
    )
    fills = adapter.execute(assessment, cycle_id="c")
    assert broker.orders == [] and fills == []


def test_hedge_is_placed_as_short(tmp_path):
    broker = FakeBroker()
    store = PositionStore(tmp_path)
    adapter = ExecutionAdapter(broker=broker, store=store)
    assessment = RiskAssessment(
        decisions=[_approved()],
        hedge=HedgeOrder(ticker="SMH", side="short", shares=5, dollars=1000.0),
    )
    adapter.execute(assessment, cycle_id="cycle-1")
    sides = {o.ticker: o.side for o in broker.orders}
    assert sides == {"NVDA": "buy", "SMH": "short"}
    smh_order = next(o for o in broker.orders if o.ticker == "SMH")
    assert smh_order.reference_price == 200.0  # dollars / shares
    assert store.get_all()  # SMH short recorded
    smh = next(p for p in store.get_all() if p.ticker == "SMH")
    assert smh.qty == -5.0


def test_rerun_is_idempotent(tmp_path):
    broker = FakeBroker()
    store = PositionStore(tmp_path)
    adapter = ExecutionAdapter(broker=broker, store=store)
    assessment = RiskAssessment(decisions=[_approved()])
    adapter.execute(assessment, cycle_id="cycle-1")
    adapter.execute(assessment, cycle_id="cycle-1")  # same cycle id
    assert store.get_all()[0].qty == 10.0  # not 20


def test_reconcile_reports_drift(tmp_path):
    class DriftBroker(FakeBroker):
        def get_positions(self):
            return [PositionRecord(ticker="NVDA", qty=8.0, avg_price=100.0)]

    store = PositionStore(tmp_path)
    adapter = ExecutionAdapter(broker=DriftBroker(), store=store)
    adapter.execute(RiskAssessment(decisions=[_approved()]), cycle_id="c")
    result = adapter.reconcile()
    assert not result.in_sync
    assert result.discrepancies[0].ticker == "NVDA"


def test_rejected_fill_does_not_update_store(tmp_path):
    class RejectBroker(FakeBroker):
        def place_order(self, order):
            self.orders.append(order)
            return Fill(
                client_order_id=order.client_order_id,
                broker_order_id="r",
                ticker=order.ticker,
                side=order.side,
                status="rejected",
                ts=datetime(2026, 6, 9, tzinfo=timezone.utc),
                reason="market closed",
            )

    store = PositionStore(tmp_path)
    adapter = ExecutionAdapter(broker=RejectBroker(), store=store)
    fills = adapter.execute(RiskAssessment(decisions=[_approved()]), cycle_id="c")
    assert fills[0].status == "rejected"
    assert store.get_all() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/execution/test_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'moneybot.execution.adapter'`

- [ ] **Step 3: Implement ExecutionAdapter**

Create `src/moneybot/execution/adapter.py`:

```python
"""ExecutionAdapter: turn a RiskAssessment into broker orders and track fills.

For each approved RiskDecision it places a buy; if a hedge is present it places a
short of the benchmark. Orders carry a deterministic client_order_id
("<cycle>:<ticker>:<side>") so a re-run never double-trades. The position store
is updated only on filled orders. reconcile() compares the store against the
broker (report-only). Exit orders (stop/target/time-stop) are the Orchestrator's
job in Plan 9 and will flow through this same adapter as sell/cover orders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from moneybot.execution.models import OrderRequest
from moneybot.execution.reconcile import reconcile

if TYPE_CHECKING:
    from moneybot.execution.broker import Broker
    from moneybot.execution.models import Fill, ReconciliationResult
    from moneybot.execution.store import PositionStore
    from moneybot.risk.models import RiskAssessment


class ExecutionAdapter:
    def __init__(self, *, broker: Broker, store: PositionStore) -> None:
        self.broker = broker
        self.store = store

    def execute(self, assessment: RiskAssessment, cycle_id: str) -> list[Fill]:
        if assessment.halted:
            return []

        fills: list[Fill] = []
        for decision in assessment.decisions:
            if not decision.approved:
                continue
            order = OrderRequest(
                client_order_id=f"{cycle_id}:{decision.ticker}:buy",
                ticker=decision.ticker,
                side="buy",
                quantity=decision.shares,
                reference_price=decision.reference_price,
            )
            fills.append(self._place(order))

        hedge = assessment.hedge
        if hedge is not None and hedge.shares > 0:
            order = OrderRequest(
                client_order_id=f"{cycle_id}:{hedge.ticker}:short",
                ticker=hedge.ticker,
                side="short",
                quantity=hedge.shares,
                reference_price=hedge.dollars / hedge.shares,
            )
            fills.append(self._place(order))

        return fills

    def _place(self, order: OrderRequest) -> Fill:
        fill = self.broker.place_order(order)
        if fill.status == "filled" and fill.filled_qty > 0:
            self.store.apply_fill(fill)
        return fill

    def reconcile(self) -> ReconciliationResult:
        return reconcile(self.store.get_all(), self.broker.get_positions())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/execution/test_adapter.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/execution/adapter.py tests/execution/test_adapter.py
git commit -m "feat(execution): ExecutionAdapter (orders, fills, store, reconcile)"
```

---

### Task 9: AlpacaBroker (thin live adapter)

The entire SDK interaction is isolated in three `_*_raw` methods returning plain dicts/lists — exactly like `YFinancePriceProvider._download`. Tests patch those methods, so no test imports the SDK or touches the network. Translation logic (our value-add) is fully tested.

**Files:**
- Create: `src/moneybot/execution/alpaca.py`
- Test: `tests/execution/test_alpaca.py`

- [ ] **Step 1: Write the failing test**

Create `tests/execution/test_alpaca.py`:

```python
from datetime import datetime, timezone

from moneybot.execution.alpaca import AlpacaBroker
from moneybot.execution.models import OrderRequest


def _clock():
    return datetime(2026, 6, 9, tzinfo=timezone.utc)


def _broker():
    return AlpacaBroker(key_id="k", secret_key="s", paper=True, clock=_clock)


def test_buy_maps_to_alpaca_buy_and_parses_fill(monkeypatch):
    b = _broker()
    captured = {}

    def fake_submit(symbol, qty, side, client_order_id):
        captured.update(symbol=symbol, qty=qty, side=side, client_order_id=client_order_id)
        return {
            "id": "abc-123",
            "status": "filled",
            "filled_qty": "10",
            "filled_avg_price": "101.5",
        }

    monkeypatch.setattr(b, "_submit_raw", fake_submit)
    order = OrderRequest(
        client_order_id="c1:NVDA:buy",
        ticker="NVDA",
        side="buy",
        quantity=10,
        reference_price=100.0,
    )
    fill = b.place_order(order)
    assert captured == {
        "symbol": "NVDA",
        "qty": 10,
        "side": "buy",
        "client_order_id": "c1:NVDA:buy",
    }
    assert fill.status == "filled"
    assert fill.broker_order_id == "abc-123"
    assert fill.filled_qty == 10 and fill.avg_price == 101.5
    assert fill.ts == _clock()


def test_short_maps_to_alpaca_sell(monkeypatch):
    b = _broker()
    captured = {}

    def fake_submit(symbol, qty, side, client_order_id):
        captured["side"] = side
        return {"id": "x", "status": "accepted", "filled_qty": "0", "filled_avg_price": None}

    monkeypatch.setattr(b, "_submit_raw", fake_submit)
    order = OrderRequest(
        client_order_id="c2:SMH:short",
        ticker="SMH",
        side="short",
        quantity=5,
        reference_price=200.0,
    )
    fill = b.place_order(order)
    assert captured["side"] == "sell"
    assert fill.status == "accepted"
    assert fill.filled_qty == 0 and fill.avg_price == 0.0


def test_rejected_status_maps_through(monkeypatch):
    b = _broker()
    monkeypatch.setattr(
        b,
        "_submit_raw",
        lambda symbol, qty, side, client_order_id: {
            "id": "x",
            "status": "rejected",
            "filled_qty": "0",
            "filled_avg_price": None,
        },
    )
    order = OrderRequest(
        client_order_id="c3", ticker="NVDA", side="buy", quantity=1, reference_price=10.0
    )
    assert b.place_order(order).status == "rejected"


def test_get_positions_parses_signed_qty(monkeypatch):
    b = _broker()
    monkeypatch.setattr(
        b,
        "_positions_raw",
        lambda: [
            {"symbol": "NVDA", "qty": "10", "avg_entry_price": "100.0"},
            {"symbol": "SMH", "qty": "-5", "avg_entry_price": "200.0"},
        ],
    )
    positions = {p.ticker: p for p in b.get_positions()}
    assert positions["NVDA"].qty == 10.0
    assert positions["SMH"].qty == -5.0 and positions["SMH"].avg_price == 200.0


def test_get_account_parses_equity_and_cash(monkeypatch):
    b = _broker()
    monkeypatch.setattr(
        b, "_account_raw", lambda: {"equity": "123456.78", "cash": "5000.00"}
    )
    acct = b.get_account()
    assert acct.equity == 123456.78 and acct.cash == 5000.0


def test_construction_does_not_touch_sdk():
    # Building the adapter must not import the SDK or open a connection — the
    # client is created lazily only inside the _*_raw methods (never called here).
    b = AlpacaBroker(key_id="k", secret_key="s", paper=True)
    assert b._client is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/execution/test_alpaca.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'moneybot.execution.alpaca'`

- [ ] **Step 3: Implement AlpacaBroker**

Create `src/moneybot/execution/alpaca.py`:

```python
"""AlpacaBroker: a thin live adapter over alpaca-py's TradingClient.

All SDK interaction is isolated in the three _*_raw methods, which return plain
dicts/lists — mirroring YFinancePriceProvider._download. Everything public is
pure translation between those primitives and moneybot's models, so tests patch
the _*_raw methods and never import the SDK or hit the network. The SDK client is
built lazily (only inside the _*_raw methods) so constructing the adapter is
free of network and import side effects.

The same TradingClient drives Alpaca's paper and live endpoints — `paper=True`
points at the paper base URL. That is what makes "paper or live by one config
flag" real. A future IBKR broker implements the same Broker Protocol; nothing
upstream changes.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from moneybot.execution.models import AccountSnapshot, Fill, PositionRecord

if TYPE_CHECKING:
    from moneybot.execution.models import OrderRequest, Side

# Our side -> Alpaca's order side. Alpaca expresses a short as a plain sell.
_ALPACA_SIDE = {"buy": "buy", "cover": "buy", "sell": "sell", "short": "sell"}

# Alpaca order status -> our Fill status.
_FILLED = {"filled"}
_REJECTED = {"rejected", "canceled", "expired"}


class AlpacaBroker:
    def __init__(
        self,
        *,
        key_id: str,
        secret_key: str,
        paper: bool = True,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._key_id = key_id
        self._secret_key = secret_key
        self._paper = paper
        self._client: Any = None  # built lazily inside _get_client
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def _get_client(self) -> Any:
        if self._client is None:
            from alpaca.trading.client import TradingClient

            self._client = TradingClient(
                self._key_id, self._secret_key, paper=self._paper
            )
        return self._client

    # --- SDK boundary: the only methods that touch alpaca-py -----------------

    def _submit_raw(
        self, symbol: str, qty: int, side: str, client_order_id: str
    ) -> dict[str, Any]:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide(side),
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        order = self._get_client().submit_order(req)
        return {
            "id": str(order.id),
            "status": str(getattr(order.status, "value", order.status)),
            "filled_qty": order.filled_qty,
            "filled_avg_price": order.filled_avg_price,
        }

    def _positions_raw(self) -> list[dict[str, Any]]:
        return [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_entry_price": p.avg_entry_price,
            }
            for p in self._get_client().get_all_positions()
        ]

    def _account_raw(self) -> dict[str, Any]:
        acct = self._get_client().get_account()
        return {"equity": acct.equity, "cash": acct.cash}

    # --- Broker Protocol: pure translation -----------------------------------

    def place_order(self, order: OrderRequest) -> Fill:
        raw = self._submit_raw(
            order.ticker, order.quantity, _ALPACA_SIDE[order.side], order.client_order_id
        )
        status = raw["status"]
        if status in _FILLED:
            our_status: Side = "filled"  # type: ignore[assignment]
        elif status in _REJECTED:
            our_status = "rejected"  # type: ignore[assignment]
        else:
            our_status = "accepted"  # type: ignore[assignment]

        return Fill(
            client_order_id=order.client_order_id,
            broker_order_id=raw["id"],
            ticker=order.ticker,
            side=order.side,
            status=our_status,  # type: ignore[arg-type]
            filled_qty=int(float(raw["filled_qty"] or 0)),
            avg_price=float(raw["filled_avg_price"] or 0.0),
            ts=self._clock(),
            reason="" if status not in _REJECTED else status,
        )

    def get_positions(self) -> list[PositionRecord]:
        return [
            PositionRecord(
                ticker=p["symbol"],
                qty=float(p["qty"]),
                avg_price=float(p["avg_entry_price"]),
            )
            for p in self._positions_raw()
        ]

    def get_account(self) -> AccountSnapshot:
        raw = self._account_raw()
        return AccountSnapshot(equity=float(raw["equity"]), cash=float(raw["cash"]))
```

Note: `status` is a plain `str` so the `Fill.status` Literal is satisfied at runtime; the `# type: ignore` comments document the intentional narrowing for static checkers (the suite does not run a type checker, but keep them for clarity).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/execution/test_alpaca.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/execution/alpaca.py tests/execution/test_alpaca.py
git commit -m "feat(execution): thin AlpacaBroker with SDK isolated behind _raw methods"
```

---

### Task 10: Factory

**Files:**
- Create: `src/moneybot/execution/factory.py`
- Test: `tests/execution/test_factory.py`

- [ ] **Step 1: Write the failing test**

Create `tests/execution/test_factory.py`:

```python
from moneybot.config import Settings
from moneybot.execution.adapter import ExecutionAdapter
from moneybot.execution.alpaca import AlpacaBroker
from moneybot.execution.factory import build_execution_adapter
from moneybot.execution.paper import PaperBroker


def test_paper_mode_builds_paper_broker(tmp_path):
    settings = Settings(mode="paper", data_dir=str(tmp_path), paper_starting_cash=50_000.0)
    adapter = build_execution_adapter(settings=settings)
    assert isinstance(adapter, ExecutionAdapter)
    assert isinstance(adapter.broker, PaperBroker)
    assert adapter.broker.cash == 50_000.0
    assert adapter.store.path == tmp_path / "positions.json"


def test_live_mode_builds_alpaca_broker(tmp_path):
    settings = Settings(
        mode="live",
        data_dir=str(tmp_path),
        alpaca_key_id="k",
        alpaca_secret_key="s",
    )
    adapter = build_execution_adapter(settings=settings)
    assert isinstance(adapter.broker, AlpacaBroker)
    assert adapter.broker._client is None  # lazy: no SDK/network on construction


def test_broker_override_is_honored(tmp_path):
    settings = Settings(mode="live", data_dir=str(tmp_path))
    sentinel = PaperBroker(starting_cash=1.0)
    adapter = build_execution_adapter(settings=settings, broker=sentinel)
    assert adapter.broker is sentinel  # override wins, no Alpaca built
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/execution/test_factory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'moneybot.execution.factory'`

- [ ] **Step 3: Implement the factory**

Create `src/moneybot/execution/factory.py`:

```python
"""Wire an ExecutionAdapter from settings.

The single `mode` flag selects the broker: paper -> PaperBroker (the validated
default), live -> AlpacaBroker. AlpacaBroker is imported lazily so paper runs
never need alpaca-py installed. A broker override short-circuits selection (used
by tests and backtests). A future IBKR broker would add one more branch here and
nothing else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from moneybot.execution.adapter import ExecutionAdapter
from moneybot.execution.paper import PaperBroker
from moneybot.execution.store import PositionStore

if TYPE_CHECKING:
    from moneybot.config import Settings
    from moneybot.execution.broker import Broker


def build_execution_adapter(
    *,
    settings: Settings,
    broker: Broker | None = None,
    store: PositionStore | None = None,
) -> ExecutionAdapter:
    if store is None:
        store = PositionStore(settings.data_dir)

    if broker is None:
        if settings.mode == "live":
            from moneybot.execution.alpaca import AlpacaBroker

            broker = AlpacaBroker(
                key_id=settings.alpaca_key_id,
                secret_key=settings.alpaca_secret_key,
                paper=False,
            )
        else:
            broker = PaperBroker(starting_cash=settings.paper_starting_cash)

    return ExecutionAdapter(broker=broker, store=store)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/execution/test_factory.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/execution/factory.py tests/execution/test_factory.py
git commit -m "feat(execution): build_execution_adapter factory (mode flag selects broker)"
```

---

### Task 11: Public exports + README

**Files:**
- Modify: `src/moneybot/execution/__init__.py`
- Modify: `README.md`
- Test: `tests/execution/test_exports.py` (Create)

- [ ] **Step 1: Write the failing test**

Create `tests/execution/test_exports.py`:

```python
import moneybot.execution as ex


def test_public_exports():
    assert ex.ExecutionAdapter is not None
    assert ex.build_execution_adapter is not None
    assert set(["ExecutionAdapter", "build_execution_adapter"]).issubset(ex.__all__)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/execution/test_exports.py -v`
Expected: FAIL — `AttributeError: module 'moneybot.execution' has no attribute 'ExecutionAdapter'`

- [ ] **Step 3: Update the package init**

Replace the contents of `src/moneybot/execution/__init__.py` with:

```python
"""Execution Adapter: places approved orders (paper or live), tracks fills,
persists positions, and reconciles against the broker. No LLM."""

from moneybot.execution.adapter import ExecutionAdapter
from moneybot.execution.factory import build_execution_adapter

__all__ = ["ExecutionAdapter", "build_execution_adapter"]
```

- [ ] **Step 4: Add the README bullet**

In `README.md`, immediately after the Phase 7 (risk engine) bullet, add:

```markdown
- **Phase 8: execution adapter** — the layer that actually places the Risk Engine's
  approved orders. One interface, paper or live by a single config flag (`mode`): a
  built-in paper-trading simulator for validation, and a thin Alpaca adapter for live.
  It records fills, keeps the bot's own position ledger, and reconciles that ledger
  against the broker (report-only — it never auto-trades to paper over a discrepancy).
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/execution/test_exports.py -v`
Expected: PASS (1 passed)

- [ ] **Step 6: Commit**

```bash
git add src/moneybot/execution/__init__.py README.md tests/execution/test_exports.py
git commit -m "feat(execution): public exports + README phase 8"
```

---

### Task 12: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS — all prior 219 tests plus the new execution tests (8 modules: config 2, models 8, positions 7, broker 2, paper 5, store 6, reconcile 6, adapter 6, alpaca 6, factory 3, exports 1 = ~52 new). No failures, no network access.

- [ ] **Step 2: Lint**

Run: `uv run ruff check src/moneybot/execution tests/execution`
Expected: no errors. If any line exceeds 100 chars (ruff's default rule set does NOT flag E501 — eyeball the longer comment lines and string literals), wrap it.

- [ ] **Step 3: Verify no network seam leaks**

Confirm no production execution module imports `alpaca` at module top level (only lazily inside methods) and no test imports `alpaca`:

Run: `uv run python -c "import ast, pathlib, sys; [print(p) for p in pathlib.Path('tests/execution').glob('*.py') if 'alpaca' in p.read_text() and 'import alpaca' in p.read_text()]"`
Expected: no output (tests reference the `AlpacaBroker` class but never `import alpaca`).

Spot-check by reading `src/moneybot/execution/alpaca.py`: every `from alpaca...` import must be inside a method body, never at module top level.

- [ ] **Step 4: Verify the single-flag switch**

Read `src/moneybot/execution/factory.py` and confirm the only thing that selects paper vs live is `settings.mode`, and that the rest of the pipeline (adapter, store, reconciliation) is broker-agnostic — confirming a future IBKR broker drops in as one new `Broker` implementation plus one factory branch.

---

## Final Self-Review (run by the controller after writing, before execution)

**1. Spec coverage (§4.6 Execution Adapter):**
- "One interface; paper or live by a single config flag" → `Broker` Protocol (Task 4) + `factory` on `settings.mode` (Task 10). ✓
- "validated code is the code that trades" → same `ExecutionAdapter`/`Broker` path for paper and live; only the leaf broker differs. ✓
- "Phase-1 broker: Alpaca or IBKR" → `AlpacaBroker` now (Task 9); IBKR is a future `Broker` impl (documented, no rework). ✓
- "order placement" → `OrderRequest` + `place_order` + adapter translation of approved decisions and hedge (Tasks 2, 5, 8, 9). ✓
- "fill tracking" → `Fill` + idempotent application to `PositionStore` (Tasks 2, 6, 8). ✓
- "reconciliation against the position store" → `PositionStore` (Task 6) + pure `reconcile` (Task 7) + `adapter.reconcile()` (Task 8). ✓
- Build-sequence step 7 ("Execution Adapter (paper) + position/fill reconciliation") fully covered; live is the thin, tested-in-shape addition the operator chose. ✓

**2. Cross-cutting invariants:**
- No network in tests — SDK isolated behind patched `_*_raw`; paper broker is in-memory. ✓
- No fabricated clock — injected `clock` everywhere a timestamp is stamped (paper, alpaca), defaulting to `datetime.now(timezone.utc)` like `JournalStore`. ✓
- No LLM in this layer. ✓
- Idempotency (real-money safety) — deterministic `client_order_id`; deduped in both `PaperBroker` and `PositionStore`. ✓
- Sign convention matches `risk.models.Position` (long +, short -). ✓
- Reconciliation is report-only — `reconcile` returns drift; nothing in the layer auto-trades to fix it. ✓

**3. Type/name consistency check:**
- `PositionRecord(ticker, qty, avg_price)` used identically by `positions.apply_fill`, `paper`, `store`, `reconcile`, `alpaca`. ✓
- `Fill` fields (`client_order_id`, `broker_order_id`, `ticker`, `side`, `status`, `filled_qty`, `avg_price`, `ts`, `reason`) consistent across producer (broker) and consumers (store, adapter, positions). ✓
- `OrderRequest.side` and `Fill.side` share the `Side` literal `("buy","sell","short","cover")`; adapter emits only `buy`/`short`; `apply_fill` and `_ALPACA_SIDE` handle all four. ✓
- `ExecutionAdapter.__init__(*, broker, store)`; `.execute(assessment, cycle_id)`; `.reconcile()` — matches factory and tests. ✓
- `build_execution_adapter(*, settings, broker=None, store=None)` — matches tests. ✓
- Consumes `RiskAssessment` (`.decisions`, `.halted`, `.hedge`), `RiskDecision` (`.approved`, `.ticker`, `.shares`, `.reference_price`), `HedgeOrder` (`.ticker`, `.shares`, `.dollars`) — all verified against `src/moneybot/risk/models.py`. ✓

No gaps found.
