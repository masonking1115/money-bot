import pytest

from moneybot.risk.metrics import average_dollar_volume, realized_volatility


def test_realized_volatility_of_constant_series_is_zero():
    assert realized_volatility([100.0, 100.0, 100.0, 100.0]) == 0.0


def test_realized_volatility_is_sample_stddev_of_returns():
    # returns: +0.10 then -0.10/1.10 ... use a hand-checkable series
    # closes 100, 110, 99 -> returns 0.10 and -0.10 -> mean 0.0, sample var = (0.01+0.01)/1 = 0.02
    vol = realized_volatility([100.0, 110.0, 99.0])
    assert vol == pytest.approx(0.02**0.5)


def test_realized_volatility_needs_at_least_three_closes():
    assert realized_volatility([100.0, 110.0]) is None
    assert realized_volatility([]) is None


def test_realized_volatility_skips_none_values():
    assert realized_volatility([100.0, None, 110.0, 99.0]) == pytest.approx(0.02**0.5)


def test_average_dollar_volume_means_price_times_volume():
    # (100*10 + 200*5) / 2 = (1000 + 1000)/2 = 1000
    assert average_dollar_volume([100.0, 200.0], [10, 5]) == 1000.0


def test_average_dollar_volume_none_when_no_pairs():
    assert average_dollar_volume([], []) is None
    assert average_dollar_volume([None], [None]) is None
