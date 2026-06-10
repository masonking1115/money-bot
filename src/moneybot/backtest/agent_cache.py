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
from typing import Literal

from moneybot.analyst.models import TradePlan
from moneybot.strategies.models import CatalystSignal


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

    def analyze(
        self, research: dict[str, list[CatalystSignal]], as_of: date | None = None
    ) -> list[TradePlan]:
        path = self._dir / f"{_key(as_of)}.json"
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            return [TradePlan.model_validate(p) for p in raw]
        if self._mode == "replay":
            raise RuntimeError(f"cache miss in replay mode for analyst as_of={as_of}")
        plans = self._inner.analyze(research, as_of=as_of)
        _atomic_write(path, json.dumps([p.model_dump(mode="json") for p in plans]))
        return plans
