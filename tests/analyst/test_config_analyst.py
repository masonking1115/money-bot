from moneybot.config import Settings


def test_analyst_settings_have_sensible_defaults():
    s = Settings()
    assert s.analyst_shortlist == 5
    assert s.rs_lookback_days == 20
    assert s.rs_timeframe == "1d"


def test_analyst_settings_override_from_kwargs():
    s = Settings(analyst_shortlist=3, rs_lookback_days=10, rs_timeframe="1h")
    assert s.analyst_shortlist == 3
    assert s.rs_lookback_days == 10
    assert s.rs_timeframe == "1h"
