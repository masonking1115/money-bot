from datetime import date

from moneybot.backtest.models import (
    BacktestConfig,
    BacktestReport,
    EquityPoint,
    PerformanceMetrics,
    TradeRecord,
)


def test_config_defaults():
    cfg = BacktestConfig(start=date(2024, 1, 1), end=date(2024, 6, 30))
    assert cfg.timeframe == "1d"
    assert cfg.starting_cash == 100_000.0
    assert cfg.mode == "record"
    assert cfg.use_agents is True


def test_config_rejects_end_before_start():
    import pytest

    with pytest.raises(ValueError):
        BacktestConfig(start=date(2024, 6, 30), end=date(2024, 1, 1))


def test_report_round_trips():
    report = BacktestReport(
        config=BacktestConfig(start=date(2024, 1, 1), end=date(2024, 1, 31)),
        equity_curve=[EquityPoint(day=date(2024, 1, 2), equity=100_000.0, cash=100_000.0, n_positions=0)],
        trades=[
            TradeRecord(
                ticker="NVDA", qty=10, entry_date=date(2024, 1, 2), exit_date=date(2024, 1, 10),
                entry_price=100.0, exit_price=110.0, pnl=100.0, pnl_pct=0.10, exit_reason="profit_target",
            )
        ],
        metrics=PerformanceMetrics(
            total_return=0.10, cagr=0.5, max_drawdown=0.05, sharpe=1.2,
            win_rate=1.0, n_trades=1, final_equity=110_000.0,
            benchmark_return=0.04, benchmark_final_equity=104_000.0,
        ),
    )
    again = BacktestReport.model_validate_json(report.model_dump_json())
    assert again.metrics.total_return == 0.10
    assert again.trades[0].ticker == "NVDA"
