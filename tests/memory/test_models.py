from datetime import datetime, timezone

from moneybot.memory.models import Dossier, JournalEntry, Lesson, MemoryContext


def test_dossier_roundtrips_json():
    d = Dossier(key="ticker:NVDA", content="# NVDA\nmoves on guidance",
                version=2, updated_at=datetime(2026, 6, 9, tzinfo=timezone.utc))
    restored = Dossier.model_validate_json(d.model_dump_json())
    assert restored == d
    assert restored.version == 2


def test_journal_entry_defaults():
    e = JournalEntry(entry_id="1", ts=datetime(2026, 6, 9, tzinfo=timezone.utc),
                     kind="proposal")
    assert e.ticker is None
    assert e.payload == {}


def test_lesson_defaults_and_fields():
    lsn = Lesson(lesson_id="1", created_at=datetime(2026, 6, 9, tzinfo=timezone.utc),
                 applies_to="sector:semiconductors", pattern="beats priced in",
                 lesson="fade post-earnings pops", confidence=0.6)
    assert lsn.evidence_trades == []
    assert lsn.supersedes is None


def test_memory_context_defaults_empty():
    ctx = MemoryContext()
    assert ctx.dossiers == []
    assert ctx.lessons == []
