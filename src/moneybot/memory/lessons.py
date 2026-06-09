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
