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
