from datetime import date

from moneybot.orchestrator.portfolio import SodEquityStore


def test_first_call_of_day_returns_zero_pnl(tmp_path):
    store = SodEquityStore(tmp_path)
    # first observation of the day anchors start-of-day equity -> 0% P&L
    assert store.day_pnl_pct(100_000.0, date(2026, 6, 10)) == 0.0


def test_same_day_computes_pnl_against_anchor(tmp_path):
    store = SodEquityStore(tmp_path)
    store.day_pnl_pct(100_000.0, date(2026, 6, 10))  # anchor
    # equity fell to 97,000 -> -3%
    assert store.day_pnl_pct(97_000.0, date(2026, 6, 10)) == -0.03


def test_anchor_persists_across_instances(tmp_path):
    SodEquityStore(tmp_path).day_pnl_pct(100_000.0, date(2026, 6, 10))
    reopened = SodEquityStore(tmp_path)
    assert reopened.day_pnl_pct(110_000.0, date(2026, 6, 10)) == 0.1


def test_new_day_reanchors(tmp_path):
    store = SodEquityStore(tmp_path)
    store.day_pnl_pct(100_000.0, date(2026, 6, 10))
    # next day, equity is 90,000 -> that becomes the new anchor -> 0%
    assert store.day_pnl_pct(90_000.0, date(2026, 6, 11)) == 0.0
    assert store.day_pnl_pct(85_500.0, date(2026, 6, 11)) == -0.05


def test_zero_anchor_is_safe(tmp_path):
    store = SodEquityStore(tmp_path)
    # a degenerate 0 anchor must not divide by zero
    assert store.day_pnl_pct(0.0, date(2026, 6, 10)) == 0.0
