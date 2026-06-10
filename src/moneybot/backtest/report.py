"""Render a BacktestReport to a terminal summary and CSV/JSON artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from moneybot.backtest.models import BacktestReport


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def render_summary(report: BacktestReport) -> str:
    m = report.metrics
    cfg = report.config
    beat = m.total_return - m.benchmark_return
    lines = [
        "=== Backtest Report ===",
        f"Period:           {cfg.start} -> {cfg.end}  ({len(report.equity_curve)} trading days)",
        f"Starting cash:    ${cfg.starting_cash:,.0f}",
        "",
        f"Total return:     {_pct(m.total_return)}   (final ${m.final_equity:,.0f})",
        f"CAGR:             {_pct(m.cagr)}",
        f"Max drawdown:     {_pct(m.max_drawdown)}",
        f"Sharpe:           {m.sharpe:.2f}",
        f"Win rate:         {_pct(m.win_rate)}   ({m.n_trades} trades)",
        "",
        f"Buy & hold SMH:   {_pct(m.benchmark_return)}   (final ${m.benchmark_final_equity:,.0f})",
        f"Strategy vs SMH:  {_pct(beat)}   ({'beat' if beat >= 0 else 'trailed'} the benchmark)",
    ]
    if report.notes:
        lines.append("")
        lines.append("Notes:")
        lines.extend(f"  - {n}" for n in report.notes)
    return "\n".join(lines)


def write_artifacts(report: BacktestReport, *, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    equity_csv = out / "equity_curve.csv"
    trades_csv = out / "trades.csv"
    report_json = out / "report.json"

    with equity_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["day", "equity", "cash", "n_positions"])
        for p in report.equity_curve:
            w.writerow([p.day.isoformat(), p.equity, p.cash, p.n_positions])

    with trades_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "ticker", "qty", "entry_date", "exit_date",
            "entry_price", "exit_price", "pnl", "pnl_pct", "exit_reason",
        ])
        for t in report.trades:
            w.writerow([
                t.ticker, t.qty, t.entry_date.isoformat(), t.exit_date.isoformat(),
                t.entry_price, t.exit_price, t.pnl, t.pnl_pct, t.exit_reason,
            ])

    report_json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return {"equity_csv": equity_csv, "trades_csv": trades_csv, "report_json": report_json}
