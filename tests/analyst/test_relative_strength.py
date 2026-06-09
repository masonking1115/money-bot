import pytest

from moneybot.analyst.relative_strength import excess_return


def test_excess_return_positive_when_name_outperforms_benchmark():
    # name +20%, benchmark +10% -> excess +10%
    rs = excess_return([100.0, 120.0], [100.0, 110.0])
    assert rs == pytest.approx(0.10)


def test_excess_return_negative_when_name_lags_benchmark():
    # name +5%, benchmark +15% -> excess -10%
    rs = excess_return([100.0, 105.0], [100.0, 115.0])
    assert rs == pytest.approx(-0.10)


def test_excess_return_uses_first_and_last_close_over_window():
    # intermediate bars don't matter; only first vs last
    rs = excess_return([100.0, 999.0, 110.0], [100.0, 1.0, 100.0])
    assert rs == pytest.approx(0.10)


def test_excess_return_zero_when_insufficient_data():
    assert excess_return([], []) == 0.0
    assert excess_return([100.0], [100.0, 110.0]) == 0.0  # name too short
    assert excess_return([100.0, 110.0], []) == 0.0       # benchmark missing


def test_excess_return_zero_when_first_close_is_zero():
    # guard against divide-by-zero / bad data
    assert excess_return([0.0, 110.0], [100.0, 110.0]) == 0.0


def test_excess_return_ignores_none_values():
    # None close values (gaps) are dropped before computing
    rs = excess_return([100.0, None, 120.0], [100.0, 110.0, None])
    assert rs == pytest.approx(0.10)
