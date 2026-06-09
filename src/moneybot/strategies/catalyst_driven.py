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
        rs = relative_strength if relative_strength is not None else {}
        scored: list[tuple[float, float, CatalystSignal]] = []
        for s in signals:
            if s.direction != "bullish":
                continue
            if s.freshness_days > self._params.freshness_window_days:
                continue
            score = self._score(s)
            if score <= 0.0:
                continue
            scored.append((score, rs.get(s.ticker, 0.0), s))

        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [
            Proposal(
                ticker=s.ticker,
                action="buy",
                conviction=s.conviction,
                thesis=s.thesis,
                score=sc,
                signal_ref=s.signal_id,
            )
            for sc, _rs, s in scored
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
