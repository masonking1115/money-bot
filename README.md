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
