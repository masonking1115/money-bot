import pytest

from moneybot.strategies import registry
from moneybot.strategies.base import Strategy
from moneybot.strategies.catalyst_driven import CatalystDrivenLong


def test_default_strategy_is_registered():
    strat = registry.get("catalyst_driven")
    assert isinstance(strat, Strategy)
    assert strat.name == "catalyst_driven"


def test_available_lists_catalyst_driven():
    assert "catalyst_driven" in registry.available()


def test_get_unknown_raises():
    with pytest.raises(KeyError, match="unknown strategy"):
        registry.get("does_not_exist")


def test_register_and_get_custom():
    sentinel = CatalystDrivenLong()
    registry.register("temp_custom", sentinel)
    try:
        assert registry.get("temp_custom") is sentinel
    finally:
        registry.unregister("temp_custom")
    assert "temp_custom" not in registry.available()
