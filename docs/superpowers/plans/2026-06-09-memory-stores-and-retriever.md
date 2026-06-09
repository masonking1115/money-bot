# Memory Stores & Retriever Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the autodidactic memory substrate — three durable stores (semantic dossiers, episodic journal, distilled lessons) plus a keyed `MemoryRetriever` that assembles the relevant slice for a set of tickers and a sector.

**Architecture:** A new `moneybot.memory` package. Three file-backed stores under a root directory: `SemanticStore` (versioned dossiers, one JSON file per key), `JournalStore` (append-only JSONL), `LessonStore` (append-only JSONL with supersede semantics). A `MemoryRetriever` protocol with a `KeyedMemoryRetriever` implementation that pulls `sector:<s>` + `ticker:<t>` dossiers and their lessons into a `MemoryContext`. All stores take an injectable `clock` so timestamps are deterministic in tests. No LLM is involved here — agents consume these in Plan 4.

**Tech Stack:** Python 3.11+, uv, pytest, pydantic v2. Pure stdlib persistence (JSON/JSONL on disk). No new dependencies.

---

## Context for the implementer

Plans 1–2 are merged on `main`. Conventions in this codebase:
- pydantic v2 models; `model_dump_json()` / `model_validate_json()` for file round-trips (handles dates/datetimes as ISO strings).
- Stores must be **durable project knowledge**, NOT the `Cache` (which is clearable). Use dedicated files under a root dir.
- For test determinism, every store takes `clock: Callable[[], datetime] | None = None`, defaulting in `__init__` to `lambda: datetime.now(timezone.utc)`. Tests pass a fixed clock.
- Keys are strings like `"sector:semiconductors"` and `"ticker:NVDA"`.
- Calibration tracking is intentionally NOT in this plan — it belongs to the learning loop (Plan 10).

---

## File Structure

- Create: `src/moneybot/memory/__init__.py` — package marker + re-exports
- Create: `src/moneybot/memory/models.py` — `Dossier`, `JournalEntry`, `Lesson`, `MemoryContext`
- Create: `src/moneybot/memory/semantic.py` — `SemanticStore`
- Create: `src/moneybot/memory/journal.py` — `JournalStore`
- Create: `src/moneybot/memory/lessons.py` — `LessonStore`
- Create: `src/moneybot/memory/retriever.py` — `MemoryRetriever` protocol + `KeyedMemoryRetriever`
- Tests: `tests/memory/__init__.py` (empty) + one test file per module
- Modify: `README.md` (status bump)

---

## Task 1: Memory models + package

**Files:**
- Create: `src/moneybot/memory/__init__.py`
- Create: `src/moneybot/memory/models.py`
- Create: `tests/memory/__init__.py` (empty)
- Test: `tests/memory/test_models.py`

- [ ] **Step 1: Create empty `tests/memory/__init__.py`** (package marker so pytest treats `tests/memory` as a package).

- [ ] **Step 2: Write the failing test `tests/memory/test_models.py`**

```python
from datetime import datetime, timezone

from moneybot.memory.models import Dossier, JournalEntry, Lesson, MemoryContext


def test_dossier_roundtrips_json():
    d = Dossier(key="ticker:NVDA", content="# NVDA\nmoves on guidance",
                version=2, updated_at=datetime(2026, 6, 9, tzinfo=timezone.utc))
    restored = Dossier.model_validate_json(d.model_dump_json())
    assert restored == d
    assert restored.version == 2


def test_journal_entry_defaults():
    e = JournalEntry(entry_id="1", ts=datetime(2026, 6, 9, tzinfo=timezone.utc),
                     kind="proposal")
    assert e.ticker is None
    assert e.payload == {}


def test_lesson_defaults_and_fields():
    lsn = Lesson(lesson_id="1", created_at=datetime(2026, 6, 9, tzinfo=timezone.utc),
                 applies_to="sector:semiconductors", pattern="beats priced in",
                 lesson="fade post-earnings pops", confidence=0.6)
    assert lsn.evidence_trades == []
    assert lsn.supersedes is None


def test_memory_context_defaults_empty():
    ctx = MemoryContext()
    assert ctx.dossiers == []
    assert ctx.lessons == []
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/memory/test_models.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'moneybot.memory'`).

- [ ] **Step 4: Write `src/moneybot/memory/models.py`**

```python
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
```

- [ ] **Step 5: Write `src/moneybot/memory/__init__.py`**

```python
"""Autodidactic memory subsystem: semantic dossiers, episodic journal, lessons."""

from moneybot.memory.models import Dossier, JournalEntry, Lesson, MemoryContext

__all__ = ["Dossier", "JournalEntry", "Lesson", "MemoryContext"]
```

- [ ] **Step 6: Run to verify it passes**

Run: `uv run pytest tests/memory/test_models.py -v`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add src/moneybot/memory/__init__.py src/moneybot/memory/models.py tests/memory/__init__.py tests/memory/test_models.py
git commit -m "feat: memory models (Dossier, JournalEntry, Lesson, MemoryContext)"
```

---

## Task 2: SemanticStore (versioned dossiers)

**Files:**
- Create: `src/moneybot/memory/semantic.py`
- Test: `tests/memory/test_semantic.py`

- [ ] **Step 1: Write the failing test `tests/memory/test_semantic.py`**

```python
from datetime import datetime, timezone

from moneybot.memory.semantic import SemanticStore

FIXED = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)


def _store(tmp_path):
    return SemanticStore(tmp_path, clock=lambda: FIXED)


def test_get_missing_returns_none(tmp_path):
    assert _store(tmp_path).get("ticker:NVDA") is None


def test_upsert_creates_version_1(tmp_path):
    store = _store(tmp_path)
    d = store.upsert("ticker:NVDA", "first")
    assert d.version == 1
    assert d.content == "first"
    assert d.updated_at == FIXED


def test_upsert_increments_version_and_replaces_content(tmp_path):
    store = _store(tmp_path)
    store.upsert("ticker:NVDA", "first")
    d2 = store.upsert("ticker:NVDA", "second")
    assert d2.version == 2
    assert d2.content == "second"
    assert store.get("ticker:NVDA").content == "second"


def test_persists_across_instances(tmp_path):
    _store(tmp_path).upsert("sector:semis", "drivers")
    assert SemanticStore(tmp_path).get("sector:semis").content == "drivers"


def test_distinct_keys_are_independent(tmp_path):
    store = _store(tmp_path)
    store.upsert("ticker:NVDA", "nv")
    store.upsert("ticker:AMD", "amd")
    assert store.get("ticker:NVDA").content == "nv"
    assert store.get("ticker:AMD").content == "amd"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/memory/test_semantic.py -v`
Expected: FAIL (`ModuleNotFoundError: ... semantic`).

- [ ] **Step 3: Write `src/moneybot/memory/semantic.py`**

```python
"""Semantic memory: versioned dossiers (sector + per-ticker knowledge).

One JSON file per key, named by a hash of the key so any key string is
filesystem-safe. Each upsert bumps the version and stamps updated_at.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from moneybot.memory.models import Dossier


class SemanticStore:
    def __init__(
        self, root: str | Path, clock: Callable[[], datetime] | None = None
    ) -> None:
        self.dir = Path(root) / "semantic"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.dir / f"{digest}.json"

    def get(self, key: str) -> Dossier | None:
        path = self._path(key)
        if not path.exists():
            return None
        return Dossier.model_validate_json(path.read_text(encoding="utf-8"))

    def upsert(self, key: str, content: str) -> Dossier:
        existing = self.get(key)
        version = existing.version + 1 if existing is not None else 1
        dossier = Dossier(
            key=key, content=content, version=version, updated_at=self._clock()
        )
        self._path(key).write_text(dossier.model_dump_json(), encoding="utf-8")
        return dossier
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/memory/test_semantic.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/memory/semantic.py tests/memory/test_semantic.py
git commit -m "feat: SemanticStore with versioned dossiers"
```

---

## Task 3: JournalStore (append-only episodic memory)

**Files:**
- Create: `src/moneybot/memory/journal.py`
- Test: `tests/memory/test_journal.py`

- [ ] **Step 1: Write the failing test `tests/memory/test_journal.py`**

```python
from datetime import date, datetime, timezone

from moneybot.memory.journal import JournalStore

FIXED = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)


def _store(tmp_path):
    return JournalStore(tmp_path, clock=lambda: FIXED)


def test_append_assigns_sequential_ids_and_clock_ts(tmp_path):
    store = _store(tmp_path)
    e1 = store.append("proposal", ticker="NVDA", payload={"action": "buy"})
    e2 = store.append("fill", ticker="NVDA")
    assert e1.entry_id == "1"
    assert e2.entry_id == "2"
    assert e1.ts == FIXED
    assert e1.payload == {"action": "buy"}
    assert e2.payload == {}


def test_read_all(tmp_path):
    store = _store(tmp_path)
    store.append("proposal", ticker="NVDA")
    store.append("proposal", ticker="AMD")
    assert len(store.read()) == 2


def test_read_filters_by_ticker_and_kind(tmp_path):
    store = _store(tmp_path)
    store.append("proposal", ticker="NVDA")
    store.append("fill", ticker="NVDA")
    store.append("proposal", ticker="AMD")
    assert len(store.read(ticker="NVDA")) == 2
    assert len(store.read(kind="proposal")) == 2
    assert len(store.read(ticker="NVDA", kind="fill")) == 1


def test_read_filters_by_since(tmp_path):
    early = JournalStore(tmp_path, clock=lambda: datetime(2026, 6, 1, tzinfo=timezone.utc))
    early.append("proposal", ticker="NVDA")
    late = JournalStore(tmp_path, clock=lambda: datetime(2026, 6, 9, tzinfo=timezone.utc))
    late.append("proposal", ticker="AMD")
    assert len(late.read(since=date(2026, 6, 5))) == 1


def test_persists_across_instances(tmp_path):
    _store(tmp_path).append("proposal", ticker="NVDA")
    assert len(JournalStore(tmp_path).read()) == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/memory/test_journal.py -v`
Expected: FAIL (`ModuleNotFoundError: ... journal`).

- [ ] **Step 3: Write `src/moneybot/memory/journal.py`**

```python
"""Episodic memory: an append-only journal of cycle events (JSONL on disk).

entry_id is a 1-based sequence (count of existing entries + 1) so it is
deterministic and ordering-stable. ts comes from the injected clock.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from moneybot.memory.models import JournalEntry


class JournalStore:
    def __init__(
        self, root: str | Path, clock: Callable[[], datetime] | None = None
    ) -> None:
        self.path = Path(root) / "journal.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def _all(self) -> list[JournalEntry]:
        if not self.path.exists():
            return []
        return [
            JournalEntry.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def append(
        self,
        kind: str,
        ticker: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> JournalEntry:
        entry = JournalEntry(
            entry_id=str(len(self._all()) + 1),
            ts=self._clock(),
            kind=kind,
            ticker=ticker,
            payload=payload or {},
        )
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")
        return entry

    def read(
        self,
        ticker: str | None = None,
        kind: str | None = None,
        since: date | None = None,
    ) -> list[JournalEntry]:
        entries = self._all()
        if ticker is not None:
            entries = [e for e in entries if e.ticker == ticker]
        if kind is not None:
            entries = [e for e in entries if e.kind == kind]
        if since is not None:
            entries = [e for e in entries if e.ts.date() >= since]
        return entries
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/memory/test_journal.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/memory/journal.py tests/memory/test_journal.py
git commit -m "feat: JournalStore append-only episodic memory"
```

---

## Task 4: LessonStore (distilled lessons with supersede)

**Files:**
- Create: `src/moneybot/memory/lessons.py`
- Test: `tests/memory/test_lessons.py`

- [ ] **Step 1: Write the failing test `tests/memory/test_lessons.py`**

```python
from datetime import datetime, timezone

from moneybot.memory.lessons import LessonStore

FIXED = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)


def _store(tmp_path):
    return LessonStore(tmp_path, clock=lambda: FIXED)


def test_add_assigns_sequential_ids_and_clock_ts(tmp_path):
    store = _store(tmp_path)
    a = store.add("ticker:NVDA", "beats priced in", "fade pops", 0.6)
    b = store.add("sector:semis", "capex cycle", "follow hyperscaler capex", 0.7)
    assert a.lesson_id == "1"
    assert b.lesson_id == "2"
    assert a.created_at == FIXED


def test_get_for_filters_by_applies_to(tmp_path):
    store = _store(tmp_path)
    store.add("ticker:NVDA", "p1", "l1", 0.5)
    store.add("ticker:AMD", "p2", "l2", 0.5)
    out = store.get_for("ticker:NVDA")
    assert [lsn.pattern for lsn in out] == ["p1"]


def test_get_for_excludes_superseded(tmp_path):
    store = _store(tmp_path)
    old = store.add("ticker:NVDA", "old", "old lesson", 0.4)
    store.add("ticker:NVDA", "new", "better lesson", 0.8, supersedes=old.lesson_id)
    out = store.get_for("ticker:NVDA")
    assert [lsn.pattern for lsn in out] == ["new"]


def test_persists_across_instances(tmp_path):
    _store(tmp_path).add("sector:semis", "p", "l", 0.5)
    assert len(LessonStore(tmp_path).get_for("sector:semis")) == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/memory/test_lessons.py -v`
Expected: FAIL (`ModuleNotFoundError: ... lessons`).

- [ ] **Step 3: Write `src/moneybot/memory/lessons.py`**

```python
"""Distilled lessons memory (JSONL on disk).

Lessons accumulate append-only. A lesson may supersede an earlier one by id;
get_for() returns only the live (non-superseded) lessons for a key.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from moneybot.memory.models import Lesson


class LessonStore:
    def __init__(
        self, root: str | Path, clock: Callable[[], datetime] | None = None
    ) -> None:
        self.path = Path(root) / "lessons.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def _all(self) -> list[Lesson]:
        if not self.path.exists():
            return []
        return [
            Lesson.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def add(
        self,
        applies_to: str,
        pattern: str,
        lesson: str,
        confidence: float,
        evidence_trades: list[str] | None = None,
        supersedes: str | None = None,
    ) -> Lesson:
        record = Lesson(
            lesson_id=str(len(self._all()) + 1),
            created_at=self._clock(),
            applies_to=applies_to,
            pattern=pattern,
            lesson=lesson,
            confidence=confidence,
            evidence_trades=evidence_trades or [],
            supersedes=supersedes,
        )
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")
        return record

    def get_for(self, applies_to: str) -> list[Lesson]:
        all_lessons = self._all()
        superseded = {lsn.supersedes for lsn in all_lessons if lsn.supersedes}
        return [
            lsn
            for lsn in all_lessons
            if lsn.applies_to == applies_to and lsn.lesson_id not in superseded
        ]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/memory/test_lessons.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/moneybot/memory/lessons.py tests/memory/test_lessons.py
git commit -m "feat: LessonStore with supersede semantics"
```

---

## Task 5: MemoryRetriever protocol + KeyedMemoryRetriever

**Files:**
- Create: `src/moneybot/memory/retriever.py`
- Modify: `src/moneybot/memory/__init__.py` (re-export the stores + retriever)
- Test: `tests/memory/test_retriever.py`

- [ ] **Step 1: Write the failing test `tests/memory/test_retriever.py`**

```python
from datetime import datetime, timezone

from moneybot.memory.lessons import LessonStore
from moneybot.memory.retriever import KeyedMemoryRetriever, MemoryRetriever
from moneybot.memory.semantic import SemanticStore

FIXED = datetime(2026, 6, 9, tzinfo=timezone.utc)


def _retriever(tmp_path):
    semantic = SemanticStore(tmp_path, clock=lambda: FIXED)
    lessons = LessonStore(tmp_path, clock=lambda: FIXED)
    semantic.upsert("sector:semis", "sector drivers")
    semantic.upsert("ticker:NVDA", "nvda dossier")
    lessons.add("sector:semis", "capex", "follow capex", 0.7)
    lessons.add("ticker:NVDA", "beats", "fade pops", 0.6)
    lessons.add("ticker:AMD", "other", "unrelated", 0.5)
    return KeyedMemoryRetriever(semantic, lessons)


def test_satisfies_protocol(tmp_path):
    assert isinstance(_retriever(tmp_path), MemoryRetriever)


def test_retrieve_gathers_sector_and_ticker_dossiers(tmp_path):
    ctx = _retriever(tmp_path).retrieve(["NVDA"], sector="semis")
    keys = {d.key for d in ctx.dossiers}
    assert keys == {"sector:semis", "ticker:NVDA"}


def test_retrieve_skips_missing_dossiers(tmp_path):
    # AMD has no dossier; it should simply be absent, not error
    ctx = _retriever(tmp_path).retrieve(["NVDA", "AMD"], sector="semis")
    keys = {d.key for d in ctx.dossiers}
    assert keys == {"sector:semis", "ticker:NVDA"}


def test_retrieve_gathers_only_relevant_lessons(tmp_path):
    ctx = _retriever(tmp_path).retrieve(["NVDA"], sector="semis")
    applies = sorted(lsn.applies_to for lsn in ctx.lessons)
    assert applies == ["sector:semis", "ticker:NVDA"]  # AMD lesson excluded
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/memory/test_retriever.py -v`
Expected: FAIL (`ModuleNotFoundError: ... retriever`).

- [ ] **Step 3: Write `src/moneybot/memory/retriever.py`**

```python
"""Memory retrieval: assemble the relevant memory slice for an agent cycle.

KeyedMemoryRetriever does deterministic key-based lookup (sector + tickers).
The MemoryRetriever protocol lets a semantic/vector backend be swapped in
later without changing callers.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from moneybot.memory.lessons import LessonStore
from moneybot.memory.models import MemoryContext
from moneybot.memory.semantic import SemanticStore


@runtime_checkable
class MemoryRetriever(Protocol):
    def retrieve(self, tickers: list[str], sector: str) -> MemoryContext:
        """Return the dossiers and lessons relevant to these tickers + sector."""
        ...


class KeyedMemoryRetriever:
    def __init__(self, semantic: SemanticStore, lessons: LessonStore) -> None:
        self.semantic = semantic
        self.lessons = lessons

    def retrieve(self, tickers: list[str], sector: str) -> MemoryContext:
        keys = [f"sector:{sector}"] + [f"ticker:{t}" for t in tickers]
        dossiers = [d for d in (self.semantic.get(k) for k in keys) if d is not None]
        lessons = [lsn for k in keys for lsn in self.lessons.get_for(k)]
        return MemoryContext(dossiers=dossiers, lessons=lessons)
```

- [ ] **Step 4: Update `src/moneybot/memory/__init__.py` to re-export stores + retriever**

```python
"""Autodidactic memory subsystem: semantic dossiers, episodic journal, lessons."""

from moneybot.memory.journal import JournalStore
from moneybot.memory.lessons import LessonStore
from moneybot.memory.models import Dossier, JournalEntry, Lesson, MemoryContext
from moneybot.memory.retriever import KeyedMemoryRetriever, MemoryRetriever
from moneybot.memory.semantic import SemanticStore

__all__ = [
    "Dossier",
    "JournalEntry",
    "Lesson",
    "MemoryContext",
    "SemanticStore",
    "JournalStore",
    "LessonStore",
    "MemoryRetriever",
    "KeyedMemoryRetriever",
]
```

- [ ] **Step 5: Run to verify it passes + full suite + lint**

Run: `uv run pytest tests/memory/test_retriever.py -v`
Expected: 4 passed.

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all tests pass; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/moneybot/memory/retriever.py src/moneybot/memory/__init__.py tests/memory/test_retriever.py
git commit -m "feat: MemoryRetriever protocol + KeyedMemoryRetriever"
```

---

## Task 6: README status bump

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the Status section of `README.md`**

Append this bullet to the existing `## Status` list (keep the Phase 1 and Phase 2 bullets):
```markdown
- Phase 3: autodidactic memory — semantic dossiers, episodic journal, distilled
  lessons, and a keyed MemoryRetriever.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README status — phase 3 memory subsystem"
```

---

## Self-Review Notes

- **Spec coverage (Plan 3 scope):** semantic dossiers (versioned) ✓ Task 2; episodic journal (append-only) ✓ Task 3; distilled lessons (with supersede) ✓ Task 4; keyed `MemoryRetriever` behind a protocol so a vector backend can replace it later ✓ Task 5. Calibration tracking is intentionally deferred to Plan 10 (learning loop), as the spec assigns it there.
- **Type consistency:** `SemanticStore.get/upsert`, `JournalStore.append/read`, `LessonStore.add/get_for`, and `KeyedMemoryRetriever.retrieve` signatures match between implementations, the retriever, and the tests. All stores share the injectable `clock` pattern. Models round-trip via `model_dump_json()`/`model_validate_json()`.
- **Determinism:** every store takes a `clock`; tests inject a fixed time. `entry_id`/`lesson_id` are sequence-based (deterministic), not random.
- **Durability vs cache:** memory is stored in dedicated files under a root dir, deliberately NOT in the clearable `Cache`.
- **No placeholders:** every step has complete, runnable code and exact commands.
