from moneybot.strategies.base import Strategy
from moneybot.strategies.catalyst_driven import CatalystDrivenLong
from moneybot.strategies.models import CatalystSignal, Evidence


def _sig(ticker, materiality, conviction, freshness_days, direction="bullish"):
    return CatalystSignal(
        ticker=ticker, category="guidance", direction=direction,
        materiality=materiality, freshness_days=freshness_days, conviction=conviction,
        evidence=[Evidence(source="edgar", quote="q", url="https://x")],
        thesis=f"{ticker} thesis", signal_id=f"sig-{ticker}",
    )


def test_satisfies_protocol():
    assert isinstance(CatalystDrivenLong(), Strategy)


def test_name_and_parameters():
    strat = CatalystDrivenLong()
    assert strat.name == "catalyst_driven"
    assert strat.parameters().freshness_window_days == 5


def test_signal_schema_is_object_with_required_fields():
    schema = CatalystDrivenLong().signal_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    for field in ("ticker", "category", "direction", "materiality",
                  "freshness_days", "conviction", "evidence", "thesis"):
        assert field in props


def test_research_guidance_mentions_semis_catalysts():
    text = CatalystDrivenLong().research_guidance().lower()
    assert "guidance" in text
    assert "export" in text  # policy catalyst


def test_rank_drops_non_bullish_and_stale():
    strat = CatalystDrivenLong()  # freshness_window_days = 5
    signals = [
        _sig("NVDA", 0.9, 0.9, 1),
        _sig("AMD", 0.9, 0.9, 1, direction="bearish"),   # dropped: not bullish
        _sig("MU", 0.9, 0.9, 9),                          # dropped: stale (>5)
    ]
    out = strat.rank(signals)
    assert [p.ticker for p in out] == ["NVDA"]
    assert out[0].action == "buy"
    assert out[0].signal_ref == "sig-NVDA"


def test_rank_orders_by_score_descending():
    strat = CatalystDrivenLong()
    signals = [
        _sig("LOW", 0.4, 0.4, 1),
        _sig("HIGH", 0.9, 0.9, 1),
        _sig("MID", 0.6, 0.6, 1),
    ]
    out = strat.rank(signals)
    assert [p.ticker for p in out] == ["HIGH", "MID", "LOW"]
    assert out[0].score > out[1].score > out[2].score


def test_rank_freshness_decay_penalizes_older_signals():
    strat = CatalystDrivenLong()
    fresh = _sig("FRESH", 0.8, 0.8, 0)
    stale = _sig("OLDER", 0.8, 0.8, 4)  # same materiality/conviction, older
    out = strat.rank([stale, fresh])
    assert [p.ticker for p in out] == ["FRESH", "OLDER"]


def test_rank_relative_strength_breaks_ties():
    strat = CatalystDrivenLong()
    a = _sig("A", 0.8, 0.8, 1)
    b = _sig("B", 0.8, 0.8, 1)
    out = strat.rank([a, b], relative_strength={"A": 0.1, "B": 0.9})
    assert [p.ticker for p in out] == ["B", "A"]


def test_exit_plan_reflects_parameters():
    plan = CatalystDrivenLong().exit_plan()
    assert plan.max_hold_days == 10
    assert plan.stop_loss_pct == 0.08
    assert plan.profit_target_pct == 0.20
    assert plan.thesis_check_guidance  # non-empty
