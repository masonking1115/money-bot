from datetime import datetime, timezone

from moneybot.memory.lessons import LessonStore
from moneybot.memory.retriever import KeyedMemoryRetriever, MemoryRetriever
from moneybot.memory.semantic import SemanticStore

FIXED = datetime(2026, 6, 9, tzinfo=timezone.utc)


def _retriever(tmp_path):
    semantic = SemanticStore(tmp_path, clock=lambda: FIXED)
    lessons = LessonStore(tmp_path, clock=lambda: FIXED)
    semantic.upsert("sector:semis", "sector drivers")
    semantic.upsert("ticker:NVDA", "nvda dossier")
    lessons.add("sector:semis", "capex", "follow capex", 0.7)
    lessons.add("ticker:NVDA", "beats", "fade pops", 0.6)
    lessons.add("ticker:AMD", "other", "unrelated", 0.5)
    return KeyedMemoryRetriever(semantic, lessons)


def test_satisfies_protocol(tmp_path):
    assert isinstance(_retriever(tmp_path), MemoryRetriever)


def test_retrieve_gathers_sector_and_ticker_dossiers(tmp_path):
    ctx = _retriever(tmp_path).retrieve(["NVDA"], sector="semis")
    keys = {d.key for d in ctx.dossiers}
    assert keys == {"sector:semis", "ticker:NVDA"}


def test_retrieve_skips_missing_dossiers(tmp_path):
    # AMD has no dossier; it should simply be absent, not error
    ctx = _retriever(tmp_path).retrieve(["NVDA", "AMD"], sector="semis")
    keys = {d.key for d in ctx.dossiers}
    assert keys == {"sector:semis", "ticker:NVDA"}


def test_retrieve_gathers_only_relevant_lessons(tmp_path):
    ctx = _retriever(tmp_path).retrieve(["NVDA"], sector="semis")
    applies = sorted(lsn.applies_to for lsn in ctx.lessons)
    assert applies == ["sector:semis", "ticker:NVDA"]  # AMD lesson excluded
