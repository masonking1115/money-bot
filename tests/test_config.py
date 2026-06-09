import textwrap

import pytest

from moneybot.config import Settings, Universe, load_universe


def test_settings_defaults_to_paper_mode(monkeypatch):
    monkeypatch.delenv("MONEYBOT_MODE", raising=False)
    settings = Settings()
    assert settings.mode == "paper"


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("MONEYBOT_MODE", "live")
    monkeypatch.setenv("MONEYBOT_DATA_DIR", "/tmp/mb-data")
    settings = Settings()
    assert settings.mode == "live"
    assert str(settings.data_dir) == "/tmp/mb-data"


def test_load_universe_parses_tickers_and_benchmark(tmp_path):
    path = tmp_path / "universe.yaml"
    path.write_text(textwrap.dedent("""
        sector: semiconductors
        benchmark: SMH
        tickers:
          - symbol: NVDA
            market_cap_tier: mega
            earnings_date: 2026-08-27
          - symbol: AMD
            market_cap_tier: large
    """))
    uni = load_universe(path)
    assert isinstance(uni, Universe)
    assert uni.sector == "semiconductors"
    assert uni.benchmark == "SMH"
    assert uni.symbols == ["NVDA", "AMD"]
    assert uni.get("NVDA").earnings_date.isoformat() == "2026-08-27"
    assert uni.get("AMD").earnings_date is None


def test_universe_get_unknown_symbol_raises(tmp_path):
    path = tmp_path / "u.yaml"
    path.write_text("sector: s\nbenchmark: B\ntickers:\n  - symbol: NVDA\n")
    uni = load_universe(path)
    with pytest.raises(KeyError):
        uni.get("TSLA")
