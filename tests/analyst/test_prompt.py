from datetime import date

from moneybot.analyst.prompt import (
    build_confirm_system,
    build_confirm_user,
    confirm_schema,
)
from moneybot.memory.models import Dossier, Lesson, MemoryContext
from moneybot.strategies.models import CatalystSignal, Evidence, Proposal


def _signal():
    return CatalystSignal(
        ticker="NVDA",
        category="guidance",
        direction="bullish",
        materiality=0.9,
        freshness_days=2,
        conviction=0.8,
        evidence=[
            Evidence(source="8-K", quote="raised FY guidance", url="https://sec/1")
        ],
        thesis="guidance raised on datacenter demand",
        signal_id="sig-1",
    )


def _proposal():
    return Proposal(
        ticker="NVDA",
        action="buy",
        conviction=0.8,
        thesis="guidance raised on datacenter demand",
        score=0.55,
        signal_ref="sig-1",
    )


def test_confirm_schema_requires_core_fields():
    schema = confirm_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    assert set(["confirmed", "adjusted_conviction", "reasoning"]).issubset(props)
    assert props["adjusted_conviction"]["minimum"] == 0
    assert props["adjusted_conviction"]["maximum"] == 1
    assert "confirmed" in schema["required"]


def test_confirm_user_includes_thesis_evidence_score_and_rs():
    user = build_confirm_user(_proposal(), _signal(), relative_strength=0.07)
    assert "NVDA" in user
    assert "guidance raised on datacenter demand" in user
    assert "raised FY guidance" in user          # evidence quote present
    assert "https://sec/1" in user                # evidence url present
    assert "0.55" in user                         # rank score present
    assert "0.07" in user                         # relative-strength reading present


def test_confirm_user_handles_missing_signal():
    # defensive: if the backing signal is absent, still produce a usable prompt
    user = build_confirm_user(_proposal(), None, relative_strength=0.0)
    assert "NVDA" in user
    assert "guidance raised on datacenter demand" in user


def test_confirm_system_includes_memory_and_ticker():
    memory = MemoryContext(
        dossiers=[
            Dossier(
                key="ticker:NVDA",
                content="NVDA moves on datacenter guidance",
                version=1,
                updated_at=date(2026, 6, 1).isoformat() + "T00:00:00+00:00",
            )
        ],
        lessons=[
            Lesson(
                lesson_id="l1",
                created_at=date(2026, 6, 1).isoformat() + "T00:00:00+00:00",
                applies_to="ticker:NVDA",
                pattern="beats priced in",
                lesson="NVDA beats are often already priced in",
                confidence=0.6,
            )
        ],
    )
    system = build_confirm_system(memory, "NVDA")
    assert "NVDA" in system
    assert "datacenter guidance" in system        # dossier content
    assert "already priced in" in system          # lesson content


def test_confirm_system_without_memory_is_clean():
    system = build_confirm_system(MemoryContext(), "NVDA")
    assert "NVDA" in system
    assert "\n\n\n" not in system                 # no triple-blank from empty memory block
