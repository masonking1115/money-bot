from datetime import date, datetime, timezone

from moneybot.memory.journal import JournalStore

FIXED = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)


def _store(tmp_path):
    return JournalStore(tmp_path, clock=lambda: FIXED)


def test_append_assigns_sequential_ids_and_clock_ts(tmp_path):
    store = _store(tmp_path)
    e1 = store.append("proposal", ticker="NVDA", payload={"action": "buy"})
    e2 = store.append("fill", ticker="NVDA")
    assert e1.entry_id == "1"
    assert e2.entry_id == "2"
    assert e1.ts == FIXED
    assert e1.payload == {"action": "buy"}
    assert e2.payload == {}


def test_read_all(tmp_path):
    store = _store(tmp_path)
    store.append("proposal", ticker="NVDA")
    store.append("proposal", ticker="AMD")
    assert len(store.read()) == 2


def test_read_filters_by_ticker_and_kind(tmp_path):
    store = _store(tmp_path)
    store.append("proposal", ticker="NVDA")
    store.append("fill", ticker="NVDA")
    store.append("proposal", ticker="AMD")
    assert len(store.read(ticker="NVDA")) == 2
    assert len(store.read(kind="proposal")) == 2
    assert len(store.read(ticker="NVDA", kind="fill")) == 1


def test_read_filters_by_since(tmp_path):
    early = JournalStore(tmp_path, clock=lambda: datetime(2026, 6, 1, tzinfo=timezone.utc))
    early.append("proposal", ticker="NVDA")
    late = JournalStore(tmp_path, clock=lambda: datetime(2026, 6, 9, tzinfo=timezone.utc))
    late.append("proposal", ticker="AMD")
    assert len(late.read(since=date(2026, 6, 5))) == 1


def test_persists_across_instances(tmp_path):
    _store(tmp_path).append("proposal", ticker="NVDA")
    assert len(JournalStore(tmp_path).read()) == 1
