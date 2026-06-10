from datetime import date

from moneybot.backtest import BacktestConfig, run_backtest  # public exports
from moneybot.backtest.__main__ import parse_args


def test_public_exports_exist():
    assert run_backtest is not None
    assert BacktestConfig is not None


def test_parse_args_minimal():
    ns = parse_args(["--start", "2024-01-01", "--end", "2024-06-30"])
    assert ns.start == date(2024, 1, 1)
    assert ns.end == date(2024, 6, 30)
    assert ns.mode == "record"        # default
    assert ns.timeframe == "1d"       # default


def test_parse_args_replay_and_cash():
    ns = parse_args(["--start", "2024-01-01", "--end", "2024-06-30", "--mode", "replay", "--cash", "50000"])
    assert ns.mode == "replay"
    assert ns.cash == 50000.0
