"""Execution Adapter: places approved orders (paper or live), tracks fills,
persists positions, and reconciles against the broker. No LLM."""

from moneybot.execution.adapter import ExecutionAdapter
from moneybot.execution.factory import build_execution_adapter

__all__ = ["ExecutionAdapter", "build_execution_adapter"]
