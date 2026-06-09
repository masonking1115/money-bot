"""Pluggable trading strategies."""

from moneybot.strategies import registry
from moneybot.strategies.base import Strategy
from moneybot.strategies.catalyst_driven import CatalystDrivenLong
from moneybot.strategies.models import (
    CatalystSignal,
    Evidence,
    ExitPlan,
    Proposal,
    StrategyParams,
)

# Register the built-in strategies on import.
registry.register(CatalystDrivenLong.name, CatalystDrivenLong())

__all__ = [
    "CatalystSignal",
    "Evidence",
    "ExitPlan",
    "Proposal",
    "StrategyParams",
    "Strategy",
    "CatalystDrivenLong",
    "registry",
]
