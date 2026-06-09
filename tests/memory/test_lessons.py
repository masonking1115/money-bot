from datetime import datetime, timezone

from moneybot.memory.lessons import LessonStore

FIXED = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)


def _store(tmp_path):
    return LessonStore(tmp_path, clock=lambda: FIXED)


def test_add_assigns_sequential_ids_and_clock_ts(tmp_path):
    store = _store(tmp_path)
    a = store.add("ticker:NVDA", "beats priced in", "fade pops", 0.6)
    b = store.add("sector:semis", "capex cycle", "follow hyperscaler capex", 0.7)
    assert a.lesson_id == "1"
    assert b.lesson_id == "2"
    assert a.created_at == FIXED


def test_get_for_filters_by_applies_to(tmp_path):
    store = _store(tmp_path)
    store.add("ticker:NVDA", "p1", "l1", 0.5)
    store.add("ticker:AMD", "p2", "l2", 0.5)
    out = store.get_for("ticker:NVDA")
    assert [lsn.pattern for lsn in out] == ["p1"]


def test_get_for_excludes_superseded(tmp_path):
    store = _store(tmp_path)
    old = store.add("ticker:NVDA", "old", "old lesson", 0.4)
    store.add("ticker:NVDA", "new", "better lesson", 0.8, supersedes=old.lesson_id)
    out = store.get_for("ticker:NVDA")
    assert [lsn.pattern for lsn in out] == ["new"]


def test_persists_across_instances(tmp_path):
    _store(tmp_path).add("sector:semis", "p", "l", 0.5)
    assert len(LessonStore(tmp_path).get_for("sector:semis")) == 1
