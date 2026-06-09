# Strategy Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pluggable strategy framework — a `Strategy` interface, a registry, and the first concrete plugin (`CatalystDrivenLong`) — so trading strategies are swappable modules the operator can experiment with later.

**Architecture:** A new `moneybot.strategies` package. Generic types (`Proposal`, `ExitPlan`, `StrategyParams`) and the catalyst signal types (`Evidence`, `CatalystSignal`) live in `models.py`. A `Strategy` protocol (`base.py`) defines the contract every strategy implements: `signal_schema`, `research_guidance`, `rank`, `exit_plan`, `parameters`. `CatalystDrivenLong` (`catalyst_driven.py`) implements it with the freshness-gated, score-ranked entry logic. A registry (`registry.py`) maps names to instances and is selected by config (`MONEYBOT_STRATEGY`). All pure logic — no LLM, no I/O — so it is fully unit-testable. Agents (later plans) become generic executors that read the active strategy.

**Tech Stack:** Python 3.11+, uv, pytest, pydantic v2. No new dependencies.

---

## Context for the implementer

Plans 1–3 are merged on `main`. The strategy spec is
`docs/superpowers/specs/2026-06-09-catalyst-driven-long-strategy-design.md` — read Sections
2.5–7 for the contract. Conventions: pydantic v2 models; `@runtime_checkable` Protocols (see
`src/moneybot/providers/__init__.py` for the established pattern); `Settings(BaseSettings)` in
`src/moneybot/config.py` (env prefix `MONEYBOT_`). This plan adds NO LLM calls — strategies are
pure decision logic that agents will later drive.

Key behavior to implement (from the spec):
- A strategy exposes a **signal schema** (JSON Schema for what Research agents extract) and
  **research guidance** (prose telling agents what catalysts to look for).
- `rank()` applies the **freshness gate** (drop catalysts older than the window, drop
  non-bullish) and ranks survivors by `materiality × conviction × freshness_decay`, with a
  relative-strength tiebreaker.
- `exit_plan()` returns the mechanical exit config (max-hold, stop-loss, profit-target) plus
  thesis-check guidance.
- `parameters()` returns the tunable defaults (Section 7 of the spec).

---

## File Structure

- Create: `src/moneybot/strategies/__init__.py` — package marker + re-exports (finalized in Task 4)
- Create: `src/moneybot/strategies/models.py` — `Evidence`, `CatalystSignal`, `Proposal`, `ExitPlan`, `StrategyParams`
- Create: `src/moneybot/strategies/base.py` — `Strategy` protocol
- Create: `src/moneybot/strategies/catalyst_driven.py` — `CatalystDrivenLong`
- Create: `src/moneybot/strategies/registry.py` — register / get / available
- Modify: `src/moneybot/config.py` — add `strategy` setting
- Modify: `README.md` — status bump
- Tests: `tests/strategies/__init__.py` (empty) + one test file per module + a config test addition

---

## Task 1: Strategy models

**Files:**
- Create: `src/moneybot/strategies/__init__.py`
- Create: `src/moneybot/strategies/models.py`
- Create: `tests/strategies/__init__.py` (empty)
- Test: `tests/strategies/test_models.py`

- [ ] **Step 1: Create empty `tests/strategies/__init__.py`** (package marker).

- [ ] **Step 2: Write the failing test `tests/strategies/test_models.py`**

```python
import pytest
from pydantic import ValidationError

from moneybot.strategies.models import (
    CatalystSignal,
    Evidence,
    ExitPlan,
    Proposal,
    StrategyParams,
)


def _signal(**over):
    base = dict(
        ticker="NVDA", category="guidance", direction="bullish",
        materiality=0.8, freshness_days=1, conviction=0.7,
        evidence=[Evidence(source="edgar", quote="raised FY guide", url="https://x/1")],
        thesis="guidance raised",
    )
    base.update(over)
    return CatalystSignal(**base)


def test_catalyst_signal_roundtrips():
    s = _signal()
    assert CatalystSignal.model_validate_json(s.model_dump_json()) == s
    assert s.signal_id is None


def test_catalyst_signal_rejects_out_of_range_conviction():
    with pytest.raises(ValidationError):
        _signal(conviction=1.5)


def test_proposal_defaults():
    p = Proposal(ticker="NVDA", action="buy", conviction=0.7, thesis="t", score=0.5)
    assert p.signal_ref is None


def test_strategy_params_defaults_match_spec():
    p = StrategyParams()
    assert p.freshness_window_days == 5
    assert p.max_hold_days == 10
    assert p.stop_loss_pct == 0.08
    assert p.profit_target_pct == 0.20
    assert p.hedge_enabled is False


def test_exit_plan_fields():
    e = ExitPlan(max_hold_days=10, stop_loss_pct=0.08, profit_target_pct=0.20,
                 thesis_check_guidance="check guidance held")
    assert e.max_hold_days == 10
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/strategies/test_models.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'moneybot.strategies'`).

- [ ] **Step 4: Write `src/moneybot/strategies/models.py`**

```python
"""Types shared across the strategy framework.

Proposal / ExitPlan / StrategyParams are generic to all strategies. Evidence /
CatalystSignal are the signal types for the catalyst-driven strategy (future
strategies may define their own signal types).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """A citation backing a catalyst claim."""

    source: str
    quote: str
    url: str


class CatalystSignal(BaseModel):
    """A structured catalyst the Research agents emit for a ticker."""

    ticker: str
    category: str
    direction: Literal["bullish", "bearish", "neutral"]
    materiality: float = Field(ge=0.0, le=1.0)
    freshness_days: int = Field(ge=0)
    conviction: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence]
    thesis: str
    signal_id: str | None = None


class Proposal(BaseModel):
    """A ranked entry candidate produced by a strategy. Sizing is the Risk
    Engine's job — the strategy only proposes the name, conviction, and score."""

    ticker: str
    action: Literal["buy"]
    conviction: float = Field(ge=0.0, le=1.0)
    thesis: str
    score: float
    signal_ref: str | None = None


class ExitPlan(BaseModel):
    """Mechanical exit configuration plus thesis-check guidance for a strategy."""

    max_hold_days: int
    stop_loss_pct: float
    profit_target_pct: float
    thesis_check_guidance: str


class StrategyParams(BaseModel):
    """Tunable parameters for a strategy (defaults are backtest starting points)."""

    freshness_window_days: int = 5
    max_hold_days: int = 10
    stop_loss_pct: float = 0.08
    profit_target_pct: float = 0.20
    max_position_pct: float = 0.10
    max_sector_exposure_pct: float = 0.60
    hedge_enabled: bool = False
```

- [ ] **Step 5: Write `src/moneybot/strategies/__init__.py`**

```python
"""Pluggable trading strategies."""

from moneybot.strategies.models import (
    CatalystSignal,
    Evidence,
    ExitPlan,
    Proposal,
    StrategyParams,
)

__all__ = [
    "CatalystSignal",
    "Evidence",
    "ExitPlan",
    "Proposal",
    "StrategyParams",
]
```

- [ ] **Step 6: Run to verify it passes**

Run: `uv run pytest tests/strategies/test_models.py -v`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add src/moneybot/strategies/__init__.py src/moneybot/strategies/models.py tests/strategies/__init__.py tests/strategies/test_models.py
git commit -m "feat: strategy framework models (CatalystSignal, Proposal, ExitPlan, StrategyParams)"
```

---

## Task 2: Strategy protocol

**Files:**
- Create: `src/moneybot/strategies/base.py`
- Test: `tests/strategies/test_base.py`

- [ ] **Step 1: Write the failing test `tests/strategies/test_base.py`**

```python
from typing import Any

from moneybot.strategies.base import Strategy
from moneybot.strategies.models import ExitPlan, Proposal, StrategyParams


class _DummyStrategy:
    name = "dummy"

    def signal_schema(self) -> dict[str, Any]:
        return {"type": "object"}

    def research_guidance(self) -> str:
        return "look for things"

    def rank(self, signals, relative_strength=None) -> list[Proposal]:
        return []

    def exit_plan(self) -> ExitPlan:
        return ExitPlan(max_hold_days=1, stop_loss_pct=0.1, profit_target_pct=0.2,
                        thesis_check_guidance="g")

    def parameters(self) -> StrategyParams:
        return StrategyParams()


def test_dummy_satisfies_protocol():
    assert isinstance(_DummyStrategy(), Strategy)


def test_non_strategy_does_not_satisfy_protocol():
    class NotAStrategy:
        name = "x"

    assert not isinstance(NotAStrategy(), Strategy)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/strategies/test_base.py -v`
Expected: FAIL (`ModuleNotFoundError: ... base`).

- [ ] **Step 3: Write `src/moneybot/strategies/base.py`**

```python
"""The Strategy contract every pluggable strategy implements."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from moneybot.strategies.models import ExitPlan, Proposal, StrategyParams


@runtime_checkable
class Strategy(Protocol):
    name: str

    def signal_schema(self) -> dict[str, Any]:
        """JSON Schema describing the signal Research agents should extract."""
        ...

    def research_guidance(self) -> str:
        """Prose telling Research agents what catalysts/patterns to look for."""
        ...

    def rank(
        self,
        signals: Sequence[Any],
        relative_strength: dict[str, float] | None = None,
    ) -> list[Proposal]:
        """Apply the strategy's entry logic and return ranked entry proposals."""
        ...

    def exit_plan(self) -> ExitPlan:
        """Mechanical exit configuration + thesis-check guidance."""
        ...

    def parameters(self) -> StrategyParams:
        """Tunable parameters (backtest-tuned)."""
        ...
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/strategies/test_base.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/strategies/base.py tests/strategies/test_base.py
git commit -m "feat: Strategy protocol"
```

---

## Task 3: CatalystDrivenLong plugin

**Files:**
- Create: `src/moneybot/strategies/catalyst_driven.py`
- Test: `tests/strategies/test_catalyst_driven.py`

- [ ] **Step 1: Write the failing test `tests/strategies/test_catalyst_driven.py`**

```python
from moneybot.strategies.base import Strategy
from moneybot.strategies.catalyst_driven import CatalystDrivenLong
from moneybot.strategies.models import CatalystSignal, Evidence


def _sig(ticker, materiality, conviction, freshness_days, direction="bullish"):
    return CatalystSignal(
        ticker=ticker, category="guidance", direction=direction,
        materiality=materiality, freshness_days=freshness_days, conviction=conviction,
        evidence=[Evidence(source="edgar", quote="q", url="https://x")],
        thesis=f"{ticker} thesis", signal_id=f"sig-{ticker}",
    )


def test_satisfies_protocol():
    assert isinstance(CatalystDrivenLong(), Strategy)


def test_name_and_parameters():
    strat = CatalystDrivenLong()
    assert strat.name == "catalyst_driven"
    assert strat.parameters().freshness_window_days == 5


def test_signal_schema_is_object_with_required_fields():
    schema = CatalystDrivenLong().signal_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    for field in ("ticker", "category", "direction", "materiality",
                  "freshness_days", "conviction", "evidence", "thesis"):
        assert field in props


def test_research_guidance_mentions_semis_catalysts():
    text = CatalystDrivenLong().research_guidance().lower()
    assert "guidance" in text
    assert "export" in text  # policy catalyst


def test_rank_drops_non_bullish_and_stale():
    strat = CatalystDrivenLong()  # freshness_window_days = 5
    signals = [
        _sig("NVDA", 0.9, 0.9, 1),
        _sig("AMD", 0.9, 0.9, 1, direction="bearish"),   # dropped: not bullish
        _sig("MU", 0.9, 0.9, 9),                          # dropped: stale (>5)
    ]
    out = strat.rank(signals)
    assert [p.ticker for p in out] == ["NVDA"]
    assert out[0].action == "buy"
    assert out[0].signal_ref == "sig-NVDA"


def test_rank_orders_by_score_descending():
    strat = CatalystDrivenLong()
    signals = [
        _sig("LOW", 0.4, 0.4, 1),
        _sig("HIGH", 0.9, 0.9, 1),
        _sig("MID", 0.6, 0.6, 1),
    ]
    out = strat.rank(signals)
    assert [p.ticker for p in out] == ["HIGH", "MID", "LOW"]
    assert out[0].score > out[1].score > out[2].score


def test_rank_freshness_decay_penalizes_older_signals():
    strat = CatalystDrivenLong()
    fresh = _sig("FRESH", 0.8, 0.8, 0)
    stale = _sig("OLDER", 0.8, 0.8, 4)  # same materiality/conviction, older
    out = strat.rank([stale, fresh])
    assert [p.ticker for p in out] == ["FRESH", "OLDER"]


def test_rank_relative_strength_breaks_ties():
    strat = CatalystDrivenLong()
    # identical scores; relative strength should order them
    a = _sig("A", 0.8, 0.8, 1)
    b = _sig("B", 0.8, 0.8, 1)
    out = strat.rank([a, b], relative_strength={"A": 0.1, "B": 0.9})
    assert [p.ticker for p in out] == ["B", "A"]


def test_exit_plan_reflects_parameters():
    plan = CatalystDrivenLong().exit_plan()
    assert plan.max_hold_days == 10
    assert plan.stop_loss_pct == 0.08
    assert plan.profit_target_pct == 0.20
    assert plan.thesis_check_guidance  # non-empty
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/strategies/test_catalyst_driven.py -v`
Expected: FAIL (`ModuleNotFoundError: ... catalyst_driven`).

- [ ] **Step 3: Write `src/moneybot/strategies/catalyst_driven.py`**

```python
"""Catalyst-Driven Long — the first strategy plugin (semiconductors, long-only).

Entry: among fresh, bullish catalysts, rank by materiality x conviction x
freshness-decay, with a relative-strength tiebreaker. Exit config is mechanical
(see exit_plan). All numbers come from StrategyParams (backtest-tuned).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from moneybot.strategies.models import (
    CatalystSignal,
    ExitPlan,
    Proposal,
    StrategyParams,
)

_RESEARCH_GUIDANCE = """\
You are reading recent filings and news for a semiconductor company to find FRESH,
material, bullish catalysts. Classify each into one of:
- guidance: an earnings beat WITH raised forward guidance (weight guidance above the
  reported quarter — it is the dominant driver in semis).
- demand: hyperscaler capex commentary, design wins, large bookings/backlog.
- supply: capacity tightening/loosening, foundry/node news, inventory normalization.
- policy: export-control changes, tariffs, subsidies.
For each catalyst, estimate materiality (0-1), conviction (0-1), and freshness in days.
Every claim MUST cite a source quote and URL; a catalyst with no citation is invalid.
Only bullish catalysts are actionable (this strategy is long-only).
"""


class CatalystDrivenLong:
    name = "catalyst_driven"

    def __init__(self, params: StrategyParams | None = None) -> None:
        self._params = params or StrategyParams()

    def parameters(self) -> StrategyParams:
        return self._params

    def signal_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "category": {
                    "type": "string",
                    "enum": ["guidance", "demand", "supply", "policy"],
                },
                "direction": {
                    "type": "string",
                    "enum": ["bullish", "bearish", "neutral"],
                },
                "materiality": {"type": "number", "minimum": 0, "maximum": 1},
                "freshness_days": {"type": "integer", "minimum": 0},
                "conviction": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "quote": {"type": "string"},
                            "url": {"type": "string"},
                        },
                        "required": ["source", "quote", "url"],
                    },
                },
                "thesis": {"type": "string"},
            },
            "required": [
                "ticker", "category", "direction", "materiality",
                "freshness_days", "conviction", "evidence", "thesis",
            ],
        }

    def research_guidance(self) -> str:
        return _RESEARCH_GUIDANCE

    def _score(self, signal: CatalystSignal) -> float:
        window = self._params.freshness_window_days
        decay = max(0.0, (window - signal.freshness_days) / window) if window else 0.0
        return signal.materiality * signal.conviction * decay

    def rank(
        self,
        signals: Sequence[CatalystSignal],
        relative_strength: dict[str, float] | None = None,
    ) -> list[Proposal]:
        rs = relative_strength or {}
        scored: list[tuple[float, float, CatalystSignal]] = []
        for s in signals:
            if s.direction != "bullish":
                continue
            if s.freshness_days > self._params.freshness_window_days:
                continue
            scored.append((self._score(s), rs.get(s.ticker, 0.0), s))

        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [
            Proposal(
                ticker=s.ticker,
                action="buy",
                conviction=s.conviction,
                thesis=s.thesis,
                score=score,
                signal_ref=s.signal_id,
            )
            for score, _rs, s in scored
        ]

    def exit_plan(self) -> ExitPlan:
        return ExitPlan(
            max_hold_days=self._params.max_hold_days,
            stop_loss_pct=self._params.stop_loss_pct,
            profit_target_pct=self._params.profit_target_pct,
            thesis_check_guidance=(
                "Re-read the latest filings/news for this name. Exit if the catalyst "
                "that justified entry has been invalidated (e.g. guidance walked back, "
                "design win lost, supply/policy reversal)."
            ),
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/strategies/test_catalyst_driven.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/strategies/catalyst_driven.py tests/strategies/test_catalyst_driven.py
git commit -m "feat: CatalystDrivenLong strategy plugin"
```

---

## Task 4: Registry + config selection

**Files:**
- Create: `src/moneybot/strategies/registry.py`
- Modify: `src/moneybot/strategies/__init__.py` (re-export registry + plugin; auto-register)
- Modify: `src/moneybot/config.py` (add `strategy` setting)
- Test: `tests/strategies/test_registry.py`
- Test: `tests/test_config.py` (add one case)

- [ ] **Step 1: Write the failing test `tests/strategies/test_registry.py`**

```python
import pytest

from moneybot.strategies import registry
from moneybot.strategies.base import Strategy
from moneybot.strategies.catalyst_driven import CatalystDrivenLong


def test_default_strategy_is_registered():
    strat = registry.get("catalyst_driven")
    assert isinstance(strat, Strategy)
    assert strat.name == "catalyst_driven"


def test_available_lists_catalyst_driven():
    assert "catalyst_driven" in registry.available()


def test_get_unknown_raises():
    with pytest.raises(KeyError, match="unknown strategy"):
        registry.get("does_not_exist")


def test_register_and_get_custom():
    sentinel = CatalystDrivenLong()
    registry.register("temp_custom", sentinel)
    try:
        assert registry.get("temp_custom") is sentinel
    finally:
        registry.unregister("temp_custom")
    assert "temp_custom" not in registry.available()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/strategies/test_registry.py -v`
Expected: FAIL (`ModuleNotFoundError: ... registry`).

- [ ] **Step 3: Write `src/moneybot/strategies/registry.py`**

```python
"""Strategy registry: map names to Strategy instances, selected by config."""

from __future__ import annotations

from moneybot.strategies.base import Strategy

_REGISTRY: dict[str, Strategy] = {}


def register(name: str, strategy: Strategy) -> None:
    _REGISTRY[name] = strategy


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def get(name: str) -> Strategy:
    if name not in _REGISTRY:
        raise KeyError(f"unknown strategy: {name!r} (available: {available()})")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)
```

- [ ] **Step 4: Update `src/moneybot/strategies/__init__.py`** (re-export + auto-register the default plugin)

```python
"""Pluggable trading strategies."""

from moneybot.strategies import registry
from moneybot.strategies.base import Strategy
from moneybot.strategies.catalyst_driven import CatalystDrivenLong
from moneybot.strategies.models import (
    CatalystSignal,
    Evidence,
    ExitPlan,
    Proposal,
    StrategyParams,
)

# Register the built-in strategies on import.
registry.register(CatalystDrivenLong.name, CatalystDrivenLong())

__all__ = [
    "CatalystSignal",
    "Evidence",
    "ExitPlan",
    "Proposal",
    "StrategyParams",
    "Strategy",
    "CatalystDrivenLong",
    "registry",
]
```

- [ ] **Step 5: Add the `strategy` setting to `src/moneybot/config.py`**

In the `Settings` class, add this field directly after the `model_analyst` line:
```python
    # Active strategy (resolved via moneybot.strategies.registry)
    strategy: str = "catalyst_driven"
```

- [ ] **Step 6: Add a config test — append to `tests/test_config.py`**

```python
def test_settings_default_strategy(monkeypatch):
    monkeypatch.delenv("MONEYBOT_STRATEGY", raising=False)
    assert Settings().strategy == "catalyst_driven"
```

- [ ] **Step 7: Run to verify all pass + full suite + lint**

Run: `uv run pytest tests/strategies/test_registry.py tests/test_config.py -v`
Expected: registry tests + config tests pass.

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all tests pass; ruff clean.

- [ ] **Step 8: Commit**

```bash
git add src/moneybot/strategies/registry.py src/moneybot/strategies/__init__.py src/moneybot/config.py tests/strategies/test_registry.py tests/test_config.py
git commit -m "feat: strategy registry + MONEYBOT_STRATEGY config selection"
```

---

## Task 5: README status bump

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Append a bullet to the `## Status` section of `README.md`**

```markdown
- Phase 4: pluggable strategy framework — Strategy interface, registry, and the
  CatalystDrivenLong plugin (semiconductors, long-only, catalyst-driven entries).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README status — phase 4 strategy framework"
```

---

## Self-Review Notes

- **Spec coverage:** Strategy interface (§2.5) ✓ Task 2; catalyst taxonomy as signal schema (§3) ✓ Task 3; entry ranking + freshness gate + RS tiebreaker (§4) ✓ Task 3 (`rank`); exit plan (§5) ✓ Task 3 (`exit_plan`); parameters/defaults (§7) ✓ Task 1 (`StrategyParams`); registry + config selection (§2.5) ✓ Task 4. Sizing/hedge enforcement (§6) is the Risk Engine's job (Plan 6), not this plan — `StrategyParams` exposes `max_position_pct`/`max_sector_exposure_pct`/`hedge_enabled` for it to consume. Dossier seeding (§8) is operator setup, not code. Thesis-invalidation *execution* and the freshness gate's interaction with live positions belong to the Analyst/orchestrator (later plans) — this plan provides the `rank`/`exit_plan` logic they call.
- **Type consistency:** `CatalystSignal`, `Evidence`, `Proposal`, `ExitPlan`, `StrategyParams` are defined in Task 1 and used identically in Tasks 2–4. `Strategy` protocol methods (`signal_schema`, `research_guidance`, `rank`, `exit_plan`, `parameters`) match between `base.py`, the dummy in `test_base.py`, and `CatalystDrivenLong`. `registry.register(name, strategy)` / `get` / `available` / `unregister` are used consistently in the tests.
- **Determinism:** all logic is pure (no clock, no I/O, no randomness); `rank` ordering is deterministic via the `(score, relative_strength)` sort key.
- **Placeholder scan:** every step has complete, runnable code and exact commands.
