from datetime import date

import pytest

from moneybot.providers import FilingsProvider
from moneybot.providers.edgar_filings import EdgarFilingsProvider


def _fake_submissions():
    return {
        "filings": {
            "recent": {
                "form": ["10-K", "8-K", "4"],
                "filingDate": ["2026-02-01", "2026-05-10", "2026-05-12"],
                "accessionNumber": [
                    "0001045810-26-000010",
                    "0001045810-26-000031",
                    "0001045810-26-000033",
                ],
                "primaryDocument": ["nvda-10k.htm", "nvda-8k.htm", "form4.xml"],
            }
        }
    }


def _provider():
    prov = EdgarFilingsProvider(cik_map={"NVDA": "0001045810"}, user_agent="test ua")
    prov._fetch_submissions = lambda cik10: _fake_submissions()
    return prov


def test_satisfies_protocol():
    assert isinstance(EdgarFilingsProvider(cik_map={}, user_agent="x"), FilingsProvider)


def test_parses_filings_oldest_first_with_urls():
    filings = _provider().get_recent_filings("NVDA")
    assert [f.form_type for f in filings] == ["10-K", "8-K", "4"]
    assert filings[0].filed_at == date(2026, 2, 1)
    assert filings[0].url == (
        "https://www.sec.gov/Archives/edgar/data/1045810/"
        "000104581026000010/nvda-10k.htm"
    )


def test_filters_by_form_type():
    filings = _provider().get_recent_filings("NVDA", types=["10-K", "8-K"])
    assert [f.form_type for f in filings] == ["10-K", "8-K"]


def test_filters_by_since():
    filings = _provider().get_recent_filings("NVDA", since=date(2026, 5, 1))
    assert [f.form_type for f in filings] == ["8-K", "4"]


def test_as_of_excludes_future_filings():
    filings = _provider().get_recent_filings("NVDA", as_of=date(2026, 5, 11))
    assert [f.filed_at for f in filings] == [date(2026, 2, 1), date(2026, 5, 10)]


def test_unknown_cik_raises():
    prov = EdgarFilingsProvider(cik_map={}, user_agent="x")
    with pytest.raises(ValueError, match="no CIK"):
        prov.get_recent_filings("TSLA")
