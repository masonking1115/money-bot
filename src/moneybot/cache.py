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
