from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from moneybot.analyst.models import TradePlan
from moneybot.config import TickerMeta, Universe
from moneybot.execution.models import AccountSnapshot, Fill, ReconciliationResult
from moneybot.orchestrator.engine import Orchestrator
from moneybot.risk.models import RiskAssessment, RiskDecision
from moneybot.strategies.models import ExitPlan

ET = ZoneInfo("America/New_York")


class FakeData:
    def __init__(self):
        self.universe = Universe(
            sector="semis", benchmark="SMH",
            tickers=[TickerMeta(symbol="NVDA"), TickerMeta(symbol="AMD")],
        )

    def get_bars(self, ticker, timeframe, lookback, as_of=None):
        return pd.DataFrame({"close": [100.0]})


class FakeBroker:
    def get_positions(self):
        return []  # no open positions -> exit phase is a no-op

    def get_account(self):
        return AccountSnapshot(equity=100_000.0, cash=100_000.0)


class FakeResearch:
    def __init__(self):
        self.called_with = None

    def research_universe(self, as_of=None):
        self.called_with = as_of
        return {"NVDA": []}


def _exit_plan():
    return ExitPlan(
        max_hold_days=10, stop_loss_pct=0.08, profit_target_pct=0.20,
        thesis_check_guidance="n/a",
    )


class FakeAnalyst:
    def analyze(self, research, as_of=None):
        return [
            TradePlan(
                ticker="NVDA", action="buy", conviction=0.7, thesis="t", score=1.0,
                exit_plan=_exit_plan(), analyst_note="ok",
            )
        ]


class FakeRisk:
    def __init__(self):
        self.portfolio = None

    def assess(self, plans, portfolio, as_of=None):
        self.portfolio = portfolio
        return RiskAssessment(
            decisions=[
                RiskDecision(
                    ticker="NVDA", approved=True, shares=10, target_dollars=1000.0,
                    target_weight=0.01, reference_price=100.0, reasoning="ok",
                )
            ]
        )


class FakeExecution:
    def __init__(self, broker):
        self.broker = broker
        self.executed = None

    def place(self, order):
        raise AssertionError("no exits expected in this test")

    def execute(self, assessment, cycle_id):
        self.executed = (assessment, cycle_id)
        return [
            Fill(
                client_order_id=f"{cycle_id}:NVDA:buy", broker_order_id="b1",
                ticker="NVDA", side="buy", status="filled", filled_qty=10,
                avg_price=100.0, ts=datetime(2026, 6, 10, tzinfo=timezone.utc),
            )
        ]

    def reconcile(self):
        return ReconciliationResult(in_sync=True, discrepancies=[])


class FakeJournal:
    def __init__(self):
        self.appended = []

    def append(self, kind, ticker=None, payload=None):
        self.appended.append((kind, ticker, payload or {}))
        return None

    def read(self, ticker=None, kind=None, since=None):
        return []


class FakeSod:
    def day_pnl_pct(self, equity, today):
        return -0.01


class FakeStrategy:
    def exit_plan(self):
        return _exit_plan()


class _Settings:
    kill_switch_file = "this-file-does-not-exist"
    risk_timeframe = "1d"
    risk_lookback_days = 20


def _orch():
    research = FakeResearch()
    risk = FakeRisk()
    execution = FakeExecution(FakeBroker())
    journal = FakeJournal()
    orch = Orchestrator(
        settings=_Settings(), data_layer=FakeData(), research=research,
        analyst=FakeAnalyst(), risk=risk, execution=execution, journal=journal,
        sod_equity=FakeSod(), strategy=FakeStrategy(),
        clock=lambda: datetime(2026, 6, 10, 10, 0, tzinfo=ET),
        market_open=lambda now: True,
    )
    return orch, research, risk, execution, journal


def test_full_cycle_runs_pipeline_and_executes_entries():
    orch, research, risk, execution, journal = _orch()
    result = orch.run_cycle()

    assert result.status == "completed" and result.cycle_id == "2026-06-10T10"
    assert result.plans_proposed == 1
    assert len(result.entry_fills) == 1 and result.entry_fills[0].ticker == "NVDA"
    assert result.reconciliation is not None and result.reconciliation.in_sync
    # risk engine saw a portfolio with the day P&L from the SOD store
    assert risk.portfolio.day_pnl_pct == -0.01
    # the execution adapter was handed the assessment + cycle id
    assert execution.executed[1] == "2026-06-10T10"
    # a buy fill was journaled with the exit plan for later time-stop lookup
    buy_journals = [p for k, t, p in journal.appended if k == "fill" and t == "NVDA"]
    assert buy_journals and buy_journals[0]["side"] == "buy"
    assert "exit_plan" in buy_journals[0]


def test_halted_assessment_sets_flag():
    research = FakeResearch()
    execution = FakeExecution(FakeBroker())

    class HaltRisk:
        def assess(self, plans, portfolio, as_of=None):
            return RiskAssessment(decisions=[], halted=True)

    orch = Orchestrator(
        settings=_Settings(), data_layer=FakeData(), research=research,
        analyst=FakeAnalyst(), risk=HaltRisk(), execution=execution,
        journal=FakeJournal(), sod_equity=FakeSod(), strategy=FakeStrategy(),
        clock=lambda: datetime(2026, 6, 10, 10, 0, tzinfo=ET),
        market_open=lambda now: True,
    )
    result = orch.run_cycle()
    assert result.status == "completed" and result.halted_by_risk is True
    assert result.entry_fills == []
