"""Wire an ExecutionAdapter from settings.

The single `mode` flag selects the broker: paper -> PaperBroker (the validated
default), live -> AlpacaBroker. AlpacaBroker is imported lazily so paper runs
never need alpaca-py installed. A broker override short-circuits selection (used
by tests and backtests). A future IBKR broker would add one more branch here and
nothing else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from moneybot.execution.adapter import ExecutionAdapter
from moneybot.execution.paper import PaperBroker
from moneybot.execution.store import PositionStore

if TYPE_CHECKING:
    from moneybot.config import Settings
    from moneybot.execution.broker import Broker


def build_execution_adapter(
    *,
    settings: Settings,
    broker: Broker | None = None,
    store: PositionStore | None = None,
) -> ExecutionAdapter:
    if store is None:
        store = PositionStore(settings.data_dir)

    if broker is None:
        if settings.mode == "live":
            from moneybot.execution.alpaca import AlpacaBroker

            broker = AlpacaBroker(
                key_id=settings.alpaca_key_id,
                secret_key=settings.alpaca_secret_key,
                paper=False,
            )
        else:
            broker = PaperBroker(starting_cash=settings.paper_starting_cash)

    return ExecutionAdapter(broker=broker, store=store)
