import pytest
from pydantic import ValidationError

from moneybot.risk.models import (
    HedgeOrder,
    PortfolioState,
    Position,
    RiskAssessment,
    RiskDecision,
)


def test_portfolio_exposure_properties():
    p = PortfolioState(
        equity=100_000.0,
        cash=40_000.0,
        positions=[
            Position(ticker="NVDA", shares=100, market_value=40_000.0),
            Position(ticker="AMD", shares=50, market_value=20_000.0),
        ],
    )
    assert p.long_market_value == 60_000.0
    assert p.gross_exposure_pct == pytest.approx(0.60)


def test_portfolio_exposure_ignores_non_positive_market_values():
    p = PortfolioState(
        equity=100_000.0,
        cash=100_000.0,
        positions=[Position(ticker="NVDA", shares=0, market_value=0.0)],
    )
    assert p.long_market_value == 0.0
    assert p.gross_exposure_pct == 0.0


def test_portfolio_requires_positive_equity():
    with pytest.raises(ValidationError):
        PortfolioState(equity=0.0, cash=0.0)


def test_assessment_approved_filters_decisions():
    approved = RiskDecision(ticker="NVDA", approved=True, target_weight=0.05,
                            target_dollars=5_000.0, shares=50, reference_price=100.0,
                            reasoning="approved")
    vetoed = RiskDecision(ticker="AMD", approved=False, rules_fired=["liquidity"],
                          reasoning="too illiquid")
    a = RiskAssessment(decisions=[approved, vetoed])
    assert a.approved == [approved]
    assert a.halted is False
    assert a.hedge is None


def test_hedge_order_is_short_only():
    h = HedgeOrder(ticker="SMH", side="short", shares=50, dollars=2_500.0)
    assert h.side == "short"
    with pytest.raises(ValidationError):
        HedgeOrder(ticker="SMH", side="long", shares=1, dollars=1.0)
