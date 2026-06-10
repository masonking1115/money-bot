from moneybot.config import Settings, TickerMeta, Universe
from moneybot.risk.engine import RiskEngine
from moneybot.risk.factory import build_risk_engine


class _Prices:
    def get_bars(self, ticker, timeframe, lookback, as_of=None):
        import pandas as pd
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])


def _datalayer(tmp_path):
    from moneybot.cache import Cache
    from moneybot.data_layer import DataLayer
    uni = Universe(sector="semiconductors", benchmark="SMH",
                   tickers=[TickerMeta(symbol="NVDA")])
    return DataLayer(uni, _Prices(), Cache(tmp_path))


def test_build_risk_engine_resolves_active_strategy(tmp_path):
    settings = Settings(strategy="catalyst_driven")
    engine = build_risk_engine(settings=settings, data_layer=_datalayer(tmp_path))
    assert isinstance(engine, RiskEngine)
    assert engine.strategy.name == "catalyst_driven"
    assert engine.settings is settings
