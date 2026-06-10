import pytest

from moneybot.risk.sizing import target_weight


def test_base_size_is_conviction_times_cap_when_vol_unknown():
    # 0.5 conviction * 0.10 cap = 0.05, no volatility info -> no scaling
    w = target_weight(conviction=0.5, volatility=None,
                      max_position_pct=0.10, target_volatility=0.02)
    assert w == pytest.approx(0.05)


def test_full_conviction_hits_the_cap():
    w = target_weight(conviction=1.0, volatility=None,
                      max_position_pct=0.10, target_volatility=0.02)
    assert w == pytest.approx(0.10)


def test_high_volatility_scales_size_down():
    # vol 0.04 vs target 0.02 -> scale 0.5 -> 0.10 * 0.5 = 0.05
    w = target_weight(conviction=1.0, volatility=0.04,
                      max_position_pct=0.10, target_volatility=0.02)
    assert w == pytest.approx(0.05)


def test_low_volatility_is_not_scaled_up_past_base():
    # calmer than target -> scale clamped to 1.0, base unchanged
    w = target_weight(conviction=1.0, volatility=0.01,
                      max_position_pct=0.10, target_volatility=0.02)
    assert w == pytest.approx(0.10)


def test_zero_or_negative_inputs_floor_at_zero():
    assert target_weight(conviction=0.0, volatility=0.02,
                         max_position_pct=0.10, target_volatility=0.02) == 0.0
    # zero volatility is treated as "no usable scaling" -> base size
    assert target_weight(conviction=0.5, volatility=0.0,
                         max_position_pct=0.10, target_volatility=0.02) == pytest.approx(0.05)
