"""Strategy registry: map names to Strategy instances, selected by config."""

from __future__ import annotations

from moneybot.strategies.base import Strategy

_REGISTRY: dict[str, Strategy] = {}


def register(name: str, strategy: Strategy) -> None:
    _REGISTRY[name] = strategy


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def get(name: str) -> Strategy:
    if name not in _REGISTRY:
        raise KeyError(f"unknown strategy: {name!r} (available: {available()})")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)
