"""CLI: python -m moneybot.backtest --start YYYY-MM-DD --end YYYY-MM-DD [--mode record|replay]

Composition root for backtests — the one place that constructs network-touching
providers. A 'record' run pays the LLM + data cost once and populates the cache;
'replay' runs are offline and free (reuse for Risk Engine parameter sweeps).

NOTE: the agent-output cache is keyed by date and assumes the research/analyst
configuration (universe, analyst_shortlist, rs_lookback_days, strategy ranking) is
unchanged. Tuning Risk Engine / exit parameters is safe against the cache; changing
research/analyst config requires a fresh --mode record run (or deleting the cache dir).
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from moneybot.backtest.engine import run_backtest
from moneybot.backtest.models import BacktestConfig
from moneybot.backtest.price_cache import CachingPriceProvider
from moneybot.backtest.report import render_summary, write_artifacts
from moneybot.cache import Cache
from moneybot.config import Settings, load_universe
from moneybot.data_layer import DataLayer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="moneybot.backtest", description="Replay history through the live bot."
    )
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--mode", choices=["record", "replay"], default="record")
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--cash", type=float, default=100_000.0)
    p.add_argument("--universe", default="universe.yaml")
    p.add_argument("--cache-dir", default="cache/backtest")
    p.add_argument("--out-dir", default="backtest_out")
    p.add_argument("--no-agents", action="store_true", help="mechanical-only: skip the AI layer")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    ns = parse_args(argv)
    settings = Settings()
    universe = load_universe(ns.universe)
    cache_root = Path(ns.cache_dir)

    # --- Real provider construction ---
    # YFinancePriceProvider: no-arg constructor; all config comes from get_bars() params.
    from moneybot.providers.yfinance_price import YFinancePriceProvider

    real_price_provider = YFinancePriceProvider()
    # Wrap with the backtest cache so record runs persist bars and replay runs are offline.
    price_provider = CachingPriceProvider(real_price_provider, root=cache_root, mode=ns.mode)

    # EdgarFilingsProvider: needs a {ticker -> CIK} map and a User-Agent string.
    # CIKs come from the universe YAML (TickerMeta.cik field); tickers with no CIK are
    # silently excluded — the research agent already degrades gracefully when filings
    # are absent for a ticker.
    from moneybot.providers.edgar_filings import EdgarFilingsProvider

    cik_map = {t.symbol: t.cik for t in universe.tickers if t.cik is not None}
    filings_provider = EdgarFilingsProvider(
        cik_map=cik_map, user_agent=settings.sec_user_agent
    )

    # RssNewsProvider: no-arg constructor; defaults to Google News RSS.
    from moneybot.providers.rss_news import RssNewsProvider

    news_provider = RssNewsProvider()

    data_layer = DataLayer(
        universe,
        price_provider,
        Cache(settings.cache_dir),
        filings_provider=filings_provider,
        news_provider=news_provider,
    )

    # KeyedMemoryRetriever: backed by SemanticStore (per-ticker dossiers) and LessonStore
    # (distilled lessons).  Both are rooted in settings.data_dir so they accumulate across
    # live-bot runs and are shared with the backtest.
    from moneybot.memory.lessons import LessonStore
    from moneybot.memory.retriever import KeyedMemoryRetriever
    from moneybot.memory.semantic import SemanticStore

    retriever = KeyedMemoryRetriever(
        semantic=SemanticStore(settings.data_dir),
        lessons=LessonStore(settings.data_dir),
    )

    llm = None  # build_research/analyst lazily construct the real Anthropic client when None

    # Trading calendar comes from the benchmark's real (cached on record) bars.
    benchmark_bars = data_layer.get_bars(
        universe.benchmark, ns.timeframe, _lookback_days(ns.start, ns.end), as_of=ns.end
    )

    config = BacktestConfig(
        start=ns.start,
        end=ns.end,
        timeframe=ns.timeframe,
        starting_cash=ns.cash,
        mode=ns.mode,
        use_agents=not ns.no_agents,
    )
    report = run_backtest(
        settings=settings,
        data_layer=data_layer,
        llm=llm,
        retriever=retriever,
        config=config,
        cache_root=cache_root,
        benchmark_bars=benchmark_bars,
    )
    print(render_summary(report))
    paths = write_artifacts(report, out_dir=ns.out_dir)
    print("\nArtifacts:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


def _lookback_days(start: date, end: date) -> int:
    # Enough calendar days to cover the range plus the risk lookback warmup.
    return (end - start).days + 60


if __name__ == "__main__":
    main()
