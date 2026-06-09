from datetime import datetime, timezone

from moneybot.memory.semantic import SemanticStore

FIXED = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)


def _store(tmp_path):
    return SemanticStore(tmp_path, clock=lambda: FIXED)


def test_get_missing_returns_none(tmp_path):
    assert _store(tmp_path).get("ticker:NVDA") is None


def test_upsert_creates_version_1(tmp_path):
    store = _store(tmp_path)
    d = store.upsert("ticker:NVDA", "first")
    assert d.version == 1
    assert d.content == "first"
    assert d.updated_at == FIXED


def test_upsert_increments_version_and_replaces_content(tmp_path):
    store = _store(tmp_path)
    store.upsert("ticker:NVDA", "first")
    d2 = store.upsert("ticker:NVDA", "second")
    assert d2.version == 2
    assert d2.content == "second"
    assert store.get("ticker:NVDA").content == "second"


def test_persists_across_instances(tmp_path):
    _store(tmp_path).upsert("sector:semis", "drivers")
    assert SemanticStore(tmp_path).get("sector:semis").content == "drivers"


def test_distinct_keys_are_independent(tmp_path):
    store = _store(tmp_path)
    store.upsert("ticker:NVDA", "nv")
    store.upsert("ticker:AMD", "amd")
    assert store.get("ticker:NVDA").content == "nv"
    assert store.get("ticker:AMD").content == "amd"
