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
