"""ExecutionAdapter: turn a RiskAssessment into broker orders and track fills.

For each approved RiskDecision it places a buy; if a hedge is present it places a
short of the benchmark. Orders carry a deterministic client_order_id
("<cycle>:<ticker>:<side>") so a re-run never double-trades. The position store
is updated only on filled orders. reconcile() compares the store against the
broker (report-only). Exit orders (stop/target/time-stop) are the Orchestrator's
job in Plan 9 and will flow through this same adapter as sell/cover orders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from moneybot.execution.models import OrderRequest
from moneybot.execution.reconcile import reconcile

if TYPE_CHECKING:
    from moneybot.execution.broker import Broker
    from moneybot.execution.models import Fill, ReconciliationResult
    from moneybot.execution.store import PositionStore
    from moneybot.risk.models import RiskAssessment


class ExecutionAdapter:
    def __init__(self, *, broker: Broker, store: PositionStore) -> None:
        self.broker = broker
        self.store = store

    def execute(self, assessment: RiskAssessment, cycle_id: str) -> list[Fill]:
        if assessment.halted:
            return []

        fills: list[Fill] = []
        for decision in assessment.decisions:
            # Skip vetoes and, defensively, any approved-but-zero-share decision
            # (the RiskEngine never emits one, but a 0 here would fail OrderRequest).
            if not decision.approved or decision.shares <= 0:
                continue
            order = OrderRequest(
                client_order_id=f"{cycle_id}:{decision.ticker}:buy",
                ticker=decision.ticker,
                side="buy",
                quantity=decision.shares,
                reference_price=decision.reference_price,
            )
            fills.append(self.place(order))

        hedge = assessment.hedge
        if hedge is not None and hedge.shares > 0:
            order = OrderRequest(
                client_order_id=f"{cycle_id}:{hedge.ticker}:short",
                ticker=hedge.ticker,
                side="short",
                quantity=hedge.shares,
                reference_price=hedge.dollars / hedge.shares,
            )
            fills.append(self.place(order))

        return fills

    def place(self, order: OrderRequest) -> Fill:
        """Place a single order and update the store on a fill.

        Used by execute() for entries and by the orchestrator for exit sells.
        """
        fill = self.broker.place_order(order)
        if fill.status == "filled" and fill.filled_qty > 0:
            self.store.apply_fill(fill)
        return fill

    def reconcile(self) -> ReconciliationResult:
        return reconcile(self.store.get_all(), self.broker.get_positions())
