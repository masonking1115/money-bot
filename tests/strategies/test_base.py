from typing import Any

from moneybot.strategies.base import Strategy
from moneybot.strategies.models import ExitPlan, Proposal, StrategyParams


class _DummyStrategy:
    name = "dummy"

    def signal_schema(self) -> dict[str, Any]:
        return {"type": "object"}

    def research_guidance(self) -> str:
        return "look for things"

    def rank(self, signals, relative_strength=None) -> list[Proposal]:
        return []

    def exit_plan(self) -> ExitPlan:
        return ExitPlan(max_hold_days=1, stop_loss_pct=0.1, profit_target_pct=0.2,
                        thesis_check_guidance="g")

    def parameters(self) -> StrategyParams:
        return StrategyParams()


def test_dummy_satisfies_protocol():
    assert isinstance(_DummyStrategy(), Strategy)


def test_non_strategy_does_not_satisfy_protocol():
    class NotAStrategy:
        name = "x"

    assert not isinstance(NotAStrategy(), Strategy)
