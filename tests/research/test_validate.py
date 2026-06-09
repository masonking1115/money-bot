from moneybot.research.validate import make_signal_id, validate_signals
from moneybot.strategies.models import CatalystSignal


def _raw(ticker="NVDA", url="https://sec/1", evidence=None, **over):
    base = {
        "ticker": ticker,
        "category": "guidance",
        "direction": "bullish",
        "materiality": 0.8,
        "freshness_days": 2,
        "conviction": 0.7,
        "evidence": evidence if evidence is not None
        else [{"source": "8-K", "quote": "Raised guidance", "url": url}],
        "thesis": "Guidance raised.",
    }
    base.update(over)
    return base


ALLOWED = {"https://sec/1", "https://news/1"}


def test_valid_signal_passes_and_is_coerced_to_model():
    out = validate_signals([_raw()], ticker="NVDA", allowed_urls=ALLOWED)
    assert len(out) == 1
    assert isinstance(out[0], CatalystSignal)
    assert out[0].signal_id is not None  # id assigned


def test_signal_with_no_evidence_is_dropped():
    out = validate_signals([_raw(evidence=[])], ticker="NVDA", allowed_urls=ALLOWED)
    assert out == []


def test_signal_citing_unknown_url_is_dropped():
    bad = _raw(evidence=[{"source": "x", "quote": "q", "url": "https://hallucinated/9"}])
    out = validate_signals([bad], ticker="NVDA", allowed_urls=ALLOWED)
    assert out == []


def test_signal_for_wrong_ticker_is_dropped():
    out = validate_signals([_raw(ticker="AMD")], ticker="NVDA", allowed_urls=ALLOWED)
    assert out == []


def test_signal_with_any_ungrounded_citation_is_dropped():
    # one good, one hallucinated url in the same signal -> drop whole signal
    mixed = _raw(evidence=[
        {"source": "8-K", "quote": "q", "url": "https://sec/1"},
        {"source": "x", "quote": "q", "url": "https://hallucinated/9"},
    ])
    out = validate_signals([mixed], ticker="NVDA", allowed_urls=ALLOWED)
    assert out == []


def test_malformed_signal_is_skipped_not_raised():
    # materiality out of range -> pydantic rejects; validator skips it
    out = validate_signals([_raw(materiality=5.0)], ticker="NVDA", allowed_urls=ALLOWED)
    assert out == []


def test_signal_id_is_deterministic_and_content_addressed():
    a = validate_signals([_raw()], ticker="NVDA", allowed_urls=ALLOWED)[0]
    b = validate_signals([_raw()], ticker="NVDA", allowed_urls=ALLOWED)[0]
    assert a.signal_id == b.signal_id  # same inputs -> same id
    # different thesis -> different id
    c = validate_signals([_raw(thesis="Different.")], ticker="NVDA", allowed_urls=ALLOWED)[0]
    assert c.signal_id != a.signal_id


def test_make_signal_id_changes_with_evidence_urls():
    s1 = CatalystSignal(ticker="NVDA", category="guidance", direction="bullish",
                        materiality=0.8, freshness_days=2, conviction=0.7,
                        evidence=[{"source": "s", "quote": "q", "url": "https://sec/1"}],
                        thesis="t")
    s2 = CatalystSignal.model_validate({
        "ticker": "NVDA", "category": "guidance", "direction": "bullish",
        "materiality": 0.8, "freshness_days": 2, "conviction": 0.7,
        "evidence": [{"source": "s", "quote": "q", "url": "https://news/1"}],
        "thesis": "t",
    })
    assert make_signal_id(s1) != make_signal_id(s2)


def test_lowercase_ticker_is_matched_and_canonicalized():
    out = validate_signals([_raw(ticker="nvda")], ticker="NVDA", allowed_urls=ALLOWED)
    assert len(out) == 1
    assert out[0].ticker == "NVDA"  # canonicalized to requested form


def test_batch_filters_independently_preserving_order():
    signals = [
        _raw(ticker="AMD"),                                    # wrong ticker -> dropped
        _raw(url="https://sec/1"),                             # valid -> kept (first)
        _raw(evidence=[]),                                     # no evidence -> dropped
        _raw(url="https://news/1", thesis="T2."),              # valid -> kept (second)
        _raw(evidence=[{"source": "x", "quote": "q",
                        "url": "https://hallucinated/9"}]),    # ungrounded -> dropped
    ]
    out = validate_signals(signals, ticker="NVDA", allowed_urls=ALLOWED)
    assert len(out) == 2
    assert out[0].thesis == "Guidance raised."   # the default _raw thesis, kept first
    assert out[1].thesis == "T2."                # kept second, order preserved
