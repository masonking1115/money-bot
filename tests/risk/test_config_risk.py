from moneybot.config import Settings


def test_risk_settings_have_sane_defaults():
    s = Settings()
    assert s.daily_loss_limit_pct == 0.03
    assert s.earnings_blackout_days == 3
    assert s.min_dollar_volume == 5_000_000.0
    assert s.target_volatility == 0.02
    assert s.hedge_ratio == 0.5
    assert s.risk_timeframe == "1d"
    assert s.risk_lookback_days == 20
    assert s.kill_switch_file == "KILL_SWITCH"


def test_risk_settings_override_from_env(monkeypatch):
    monkeypatch.setenv("MONEYBOT_DAILY_LOSS_LIMIT_PCT", "0.05")
    monkeypatch.setenv("MONEYBOT_RISK_LOOKBACK_DAYS", "30")
    s = Settings()
    assert s.daily_loss_limit_pct == 0.05
    assert s.risk_lookback_days == 30
