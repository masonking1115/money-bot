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
