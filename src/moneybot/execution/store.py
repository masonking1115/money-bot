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
