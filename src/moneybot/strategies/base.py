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
