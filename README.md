# money-bot

AI sector-specialist trading bot. Agents interpret unstructured data (filings, news);
deterministic code handles risk and execution. Paper-first; live by a single config flag.

See the design spec in `docs/superpowers/specs/` and plans in `docs/superpowers/plans/`.

## Setup

```bash
uv sync
cp .env.example .env            # fill in as you add providers
cp universe.example.yaml universe.yaml   # edit to your sector
uv run pytest -q
```

## Status

- Phase 1: foundation + point-in-time price data layer.
- Phase 2: filings (SEC EDGAR), news (RSS), and fundamentals (yfinance) providers,
  integrated into the DataLayer with caching and point-in-time access.
- Phase 3: autodidactic memory — semantic dossiers, episodic journal, distilled
  lessons, and a keyed MemoryRetriever.
- Phase 4: pluggable strategy framework — Strategy interface, registry, and the
  CatalystDrivenLong plugin (semiconductors, long-only, catalyst-driven entries).
- Phase 5: research agents — generic tiered LLM agents (Haiku triage → Sonnet
  deep-read) that read the active strategy's signal schema + research guidance, pull
  filings/news (point-in-time aware) plus operator memory, and emit citation-grounded
  CatalystSignals. All Anthropic calls sit behind an LLMClient seam, so no test touches
  the network.
- Phase 6: analyst agent — a generic, strategy-agnostic agent (moneybot.analyst) that turns
  raw research signals into ranked, independently-confirmed TradePlans. It computes each name's
  relative strength vs the benchmark, delegates the freshness gate + ranking to the active
  strategy's rank, then makes one Opus call per shortlisted name to independently confirm or
  reject the thesis (a malformed response is a safe rejection — never an unverified trade).
  Confirmed proposals carry the strategy's exit plan and the analyst's adjusted conviction. The
  Analyst proposes; it never sizes or places trades (that is the Risk Engine, Phase 7).
- Phase 7: risk engine — a deterministic, pure-Python layer (moneybot.risk) the agents
  cannot bypass. It takes the Analyst's TradePlans plus a portfolio snapshot and approves,
  downsizes, or vetoes each against hard limits: a kill switch and daily-loss circuit
  breaker halt all new entries; per name it blocks pyramiding and earnings-window entries,
  checks liquidity and price sanity, then sizes by conviction scaled down for volatility,
  bounded by per-name, sector-exposure, and cash caps. Every decision records the rule that
  fired, and an optional SMH hedge offsets sector beta when enabled. No LLM, no network.
- Phase 8: execution adapter — the layer (moneybot.execution) that actually places the Risk
  Engine's approved orders. One interface, paper or live by a single config flag (`mode`): a
  built-in paper-trading simulator for validation, and a thin Alpaca adapter for live (its SDK
  calls isolated behind a seam, so no test touches the network). It records fills, keeps the
  bot's own position ledger (idempotent, crash-safe), and reconciles that ledger against the
  broker — report-only, never auto-trading to paper over a discrepancy. A future broker (e.g.
  IBKR) is just one more implementation of the same Broker seam.
- Phase 9: orchestrator — the conductor (moneybot.orchestrator) that runs one full trading
  cycle end-to-end: it checks the kill switch and market hours, closes any positions that hit
  their stop-loss / profit-target / time-stop, then runs research → analyst → a marked-to-market
  portfolio snapshot → risk engine → entry execution, and journals every step before reconciling
  against the broker. Every component is injected, so the whole cycle runs in tests with fakes —
  no network, no LLM, an injected clock. `build_orchestrator` wires the entire bot from settings.

### Phase 10 — Backtesting harness

The backtester replays historical market data through the *exact same* code the bot
runs live — research → analyst → risk → execution → exits — one simulated trading day
at a time, with point-in-time data so it can never peek at the future. It's how we
check whether the strategy actually has an edge before risking real money.

- **Record once, replay free.** The first run ("record" mode) pays for the AI work
  (the Claude research + analyst calls) and the data downloads, and caches the AI's
  per-day decisions and the prices to disk. Every later run ("replay" mode) reuses
  that cache — no Claude calls, fully offline — so you can sweep Risk Engine settings
  (position size, stop-loss, profit target, exposure caps) cheaply and instantly.
  The cache is keyed by date and assumes the research/analyst setup is unchanged;
  if you change the universe or analyst settings, re-run in record mode.
- **Daily cadence.** Free price history is deep at daily resolution but only weeks
  deep intraday, so the backtest runs one cycle per trading day (trading days come
  from the sector ETF's real bar dates, so holidays are handled automatically).
- **What you get.** An equity curve (marked to market each day), total return, max
  drawdown, Sharpe, win rate, trade count, and a side-by-side comparison against just
  buying and holding the sector ETF (SMH) — printed as a summary and written as CSV +
  JSON for deeper analysis.
- **One limitation to know:** the intraday daily-loss circuit breaker can't be
  exercised by a daily backtest (with one cycle per day there's no intraday move to
  trip it). It still protects live/paper trading; the backtest just can't test it.

Run it:

```bash
# First time (pays LLM + download cost, populates the cache):
uv run python -m moneybot.backtest --start 2024-01-01 --end 2024-12-31 --mode record

# Re-run after changing Risk Engine settings (free, offline):
uv run python -m moneybot.backtest --start 2024-01-01 --end 2024-12-31 --mode replay
```
