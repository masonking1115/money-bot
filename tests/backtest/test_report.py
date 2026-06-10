from datetime import date

from moneybot.backtest.models import (
    BacktestConfig,
    BacktestReport,
    EquityPoint,
    PerformanceMetrics,
    TradeRecord,
)
from moneybot.backtest.report import render_summary, write_artifacts


def _report():
    return BacktestReport(
        config=BacktestConfig(start=date(2024, 1, 2), end=date(2024, 1, 31)),
        equity_curve=[
            EquityPoint(day=date(2024, 1, 2), equity=100_000.0, cash=100_000.0, n_positions=0),
            EquityPoint(day=date(2024, 1, 31), equity=112_000.0, cash=5_000.0, n_positions=2),
        ],
        trades=[
            TradeRecord(
                ticker="NVDA", qty=10, entry_date=date(2024, 1, 2), exit_date=date(2024, 1, 20),
                entry_price=100.0, exit_price=120.0, pnl=200.0, pnl_pct=0.20, exit_reason="profit_target",
            )
        ],
        metrics=PerformanceMetrics(
            total_return=0.12, cagr=2.0, max_drawdown=0.04, sharpe=1.5,
            win_rate=1.0, n_trades=1, final_equity=112_000.0,
            benchmark_return=0.05, benchmark_final_equity=105_000.0,
        ),
        notes=["daily-loss breaker note"],
    )


def test_render_summary_mentions_headline_numbers_and_benchmark():
    text = render_summary(_report())
    assert "Total return" in text
    assert "12.00%" in text          # strategy return formatted as percent
    assert "5.00%" in text           # benchmark return
    assert "SMH" in text or "benchmark" in text.lower()
    assert "Max drawdown" in text
    assert "daily-loss breaker note" in text


def test_write_artifacts_creates_files(tmp_path):
    paths = write_artifacts(_report(), out_dir=tmp_path)
    assert paths["equity_csv"].exists()
    assert paths["trades_csv"].exists()
    assert paths["report_json"].exists()
    equity_text = paths["equity_csv"].read_text(encoding="utf-8")
    assert "day,equity,cash,n_positions" in equity_text
    assert "2024-01-31" in equity_text
    trades_text = paths["trades_csv"].read_text(encoding="utf-8")
    assert "NVDA" in trades_text
    # JSON round-trips back into the model
    BacktestReport.model_validate_json(paths["report_json"].read_text(encoding="utf-8"))
