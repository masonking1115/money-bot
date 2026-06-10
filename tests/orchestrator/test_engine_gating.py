from datetime import datetime
from zoneinfo import ZoneInfo

from moneybot.orchestrator.engine import Orchestrator

ET = ZoneInfo("America/New_York")


class FakeJournal:
    def __init__(self):
        self.entries = []

    def append(self, kind, ticker=None, payload=None):
        self.entries.append((kind, ticker, payload or {}))
        return None

    def read(self, ticker=None, kind=None, since=None):
        return []


class _Settings:
    kill_switch_file = "this-file-does-not-exist"
    data_dir = "."
    risk_timeframe = "1d"
    risk_lookback_days = 20


def _orch(*, clock, market_open, journal=None, **kw):
    # Collaborators that must NOT be called on the gated paths get None;
    # if the gate is wrong, the test crashes on a None attribute access.
    return Orchestrator(
        settings=_Settings(),
        data_layer=kw.get("data_layer"),
        research=kw.get("research"),
        analyst=kw.get("analyst"),
        risk=kw.get("risk"),
        execution=kw.get("execution"),
        journal=journal or FakeJournal(),
        sod_equity=kw.get("sod_equity"),
        strategy=kw.get("strategy"),
        clock=clock,
        market_open=market_open,
    )


def test_kill_switch_halts(monkeypatch):
    monkeypatch.setenv("MONEYBOT_KILL_SWITCH", "1")
    journal = FakeJournal()
    orch = _orch(
        clock=lambda: datetime(2026, 6, 10, 10, 0, tzinfo=ET),
        market_open=lambda now: True,
        journal=journal,
    )
    result = orch.run_cycle()
    assert result.status == "halted" and result.reason == "kill_switch"
    assert ("halt", None, {"reason": "kill_switch"}) in journal.entries


def test_market_closed_skips():
    journal = FakeJournal()
    orch = _orch(
        clock=lambda: datetime(2026, 6, 13, 12, 0, tzinfo=ET),  # Saturday
        market_open=lambda now: False,
        journal=journal,
    )
    result = orch.run_cycle()
    assert result.status == "skipped" and result.reason == "market_closed"


def test_cycle_id_is_derived_from_clock():
    orch = _orch(
        clock=lambda: datetime(2026, 6, 10, 14, 0, tzinfo=ET),
        market_open=lambda now: False,  # skip early so no collaborators run
    )
    result = orch.run_cycle()
    assert result.cycle_id == "2026-06-10T14"
