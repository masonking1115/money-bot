from datetime import datetime, timezone

from moneybot.execution.adapter import ExecutionAdapter
from moneybot.execution.models import AccountSnapshot, Fill, PositionRecord
from moneybot.execution.store import PositionStore
from moneybot.risk.models import HedgeOrder, RiskAssessment, RiskDecision


class FakeBroker:
    """Records placed orders; fills buys/shorts at their reference price."""

    def __init__(self):
        self.orders = []
        self._fills = {}  # client_order_id -> Fill (idempotent, like a real broker)
        self._seq = 0

    def place_order(self, order):
        prior = self._fills.get(order.client_order_id)
        if prior is not None:
            return prior  # repeated order_id: return the original fill, do not re-trade
        self.orders.append(order)
        self._seq += 1
        fill = Fill(
            client_order_id=order.client_order_id,
            broker_order_id=f"fake-{self._seq}",
            ticker=order.ticker,
            side=order.side,
            status="filled",
            filled_qty=order.quantity,
            avg_price=order.reference_price,
            ts=datetime(2026, 6, 9, tzinfo=timezone.utc),
        )
        self._fills[order.client_order_id] = fill
        return fill

    def get_positions(self):
        return []

    def get_account(self):
        return AccountSnapshot(equity=100_000.0, cash=100_000.0)


def _approved(ticker="NVDA", shares=10, price=100.0):
    return RiskDecision(
        ticker=ticker,
        approved=True,
        target_weight=0.01,
        target_dollars=shares * price,
        shares=shares,
        reference_price=price,
        reasoning="approved within limits",
    )


def _vetoed(ticker="AMD"):
    return RiskDecision(
        ticker=ticker, approved=False, rules_fired=["liquidity"], reasoning="thin"
    )


def test_execute_places_approved_orders_and_updates_store(tmp_path):
    broker = FakeBroker()
    store = PositionStore(tmp_path)
    adapter = ExecutionAdapter(broker=broker, store=store)

    assessment = RiskAssessment(decisions=[_approved(), _vetoed()])
    fills = adapter.execute(assessment, cycle_id="cycle-1")

    assert len(broker.orders) == 1  # only the approved one
    assert broker.orders[0].client_order_id == "cycle-1:NVDA:buy"
    assert broker.orders[0].side == "buy"
    assert len(fills) == 1 and fills[0].status == "filled"
    assert store.get_all()[0].ticker == "NVDA" and store.get_all()[0].qty == 10.0


def test_approved_with_zero_shares_is_skipped(tmp_path):
    # The RiskEngine never emits this, but guard against quantity=0 reaching OrderRequest.
    broker = FakeBroker()
    store = PositionStore(tmp_path)
    adapter = ExecutionAdapter(broker=broker, store=store)
    bad = RiskDecision(
        ticker="NVDA", approved=True, shares=0, reference_price=100.0, reasoning="zero"
    )
    fills = adapter.execute(RiskAssessment(decisions=[bad]), cycle_id="c")
    assert fills == [] and broker.orders == [] and store.get_all() == []


def test_halted_assessment_places_nothing(tmp_path):
    broker = FakeBroker()
    adapter = ExecutionAdapter(broker=broker, store=PositionStore(tmp_path))
    assessment = RiskAssessment(
        decisions=[_vetoed("NVDA")], halted=True
    )
    fills = adapter.execute(assessment, cycle_id="c")
    assert broker.orders == [] and fills == []


def test_hedge_is_placed_as_short(tmp_path):
    broker = FakeBroker()
    store = PositionStore(tmp_path)
    adapter = ExecutionAdapter(broker=broker, store=store)
    assessment = RiskAssessment(
        decisions=[_approved()],
        hedge=HedgeOrder(ticker="SMH", side="short", shares=5, dollars=1000.0),
    )
    adapter.execute(assessment, cycle_id="cycle-1")
    sides = {o.ticker: o.side for o in broker.orders}
    assert sides == {"NVDA": "buy", "SMH": "short"}
    smh_order = next(o for o in broker.orders if o.ticker == "SMH")
    assert smh_order.reference_price == 200.0  # dollars / shares
    assert store.get_all()  # SMH short recorded
    smh = next(p for p in store.get_all() if p.ticker == "SMH")
    assert smh.qty == -5.0


def test_rerun_is_idempotent(tmp_path):
    broker = FakeBroker()
    store = PositionStore(tmp_path)
    adapter = ExecutionAdapter(broker=broker, store=store)
    assessment = RiskAssessment(decisions=[_approved()])
    adapter.execute(assessment, cycle_id="cycle-1")
    adapter.execute(assessment, cycle_id="cycle-1")  # same cycle id
    assert store.get_all()[0].qty == 10.0  # not 20
    assert len(broker.orders) == 1  # broker saw the order once; the rerun was a no-op


def test_reconcile_reports_drift(tmp_path):
    class DriftBroker(FakeBroker):
        def get_positions(self):
            return [PositionRecord(ticker="NVDA", qty=8.0, avg_price=100.0)]

    store = PositionStore(tmp_path)
    adapter = ExecutionAdapter(broker=DriftBroker(), store=store)
    adapter.execute(RiskAssessment(decisions=[_approved()]), cycle_id="c")
    result = adapter.reconcile()
    assert not result.in_sync
    assert result.discrepancies[0].ticker == "NVDA"


def test_rejected_fill_does_not_update_store(tmp_path):
    class RejectBroker(FakeBroker):
        def place_order(self, order):
            self.orders.append(order)
            return Fill(
                client_order_id=order.client_order_id,
                broker_order_id="r",
                ticker=order.ticker,
                side=order.side,
                status="rejected",
                ts=datetime(2026, 6, 9, tzinfo=timezone.utc),
                reason="market closed",
            )

    store = PositionStore(tmp_path)
    adapter = ExecutionAdapter(broker=RejectBroker(), store=store)
    fills = adapter.execute(RiskAssessment(decisions=[_approved()]), cycle_id="c")
    assert fills[0].status == "rejected"
    assert store.get_all() == []


def test_place_sell_updates_store(tmp_path):
    from moneybot.execution.models import OrderRequest

    broker = FakeBroker()
    store = PositionStore(tmp_path)
    adapter = ExecutionAdapter(broker=broker, store=store)
    # open a long first
    adapter.execute(RiskAssessment(decisions=[_approved(shares=10, price=100.0)]), cycle_id="c1")
    # now place a direct sell of the whole position
    sell = OrderRequest(
        client_order_id="c2:NVDA:exit",
        ticker="NVDA",
        side="sell",
        quantity=10,
        reference_price=130.0,
    )
    fill = adapter.place(sell)
    assert fill.status == "filled" and fill.side == "sell"
    assert store.get_all() == []  # position closed
