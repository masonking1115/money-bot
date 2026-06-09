from datetime import date

from moneybot.providers import FundamentalsProvider
from moneybot.providers.yfinance_fundamentals import YFinanceFundamentalsProvider


def _provider():
    prov = YFinanceFundamentalsProvider()
    prov._fetch_info = lambda ticker: {
        "marketCap": 3_000_000_000_000,
        "trailingPE": 55.5,
        "totalRevenue": 130_000_000_000,
        "irrelevant": "ignored",
    }
    return prov


def test_satisfies_protocol():
    assert isinstance(YFinanceFundamentalsProvider(), FundamentalsProvider)


def test_maps_info_fields():
    fund = _provider().get_fundamentals("NVDA", as_of=date(2026, 6, 9))
    assert fund.ticker == "NVDA"
    assert fund.as_of == date(2026, 6, 9)
    assert fund.market_cap == 3_000_000_000_000
    assert fund.pe_ratio == 55.5
    assert fund.revenue == 130_000_000_000


def test_missing_fields_become_none():
    prov = YFinanceFundamentalsProvider()
    prov._fetch_info = lambda ticker: {}
    fund = prov.get_fundamentals("AMD", as_of=date(2026, 6, 9))
    assert fund.market_cap is None
    assert fund.pe_ratio is None
    assert fund.revenue is None
