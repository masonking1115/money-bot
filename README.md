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
- **Phase 6 — Analyst agent:** a generic, strategy-agnostic agent (`moneybot.analyst`) that turns
  raw research signals into ranked, independently-confirmed `TradePlan`s. It computes each name's
  relative strength vs the benchmark, delegates the freshness gate + ranking to the active
  strategy's `rank`, then makes one Opus call per shortlisted name to *independently confirm or
  reject* the thesis (a malformed response is a safe rejection — never an unverified trade).
  Confirmed proposals carry the strategy's exit plan and the analyst's adjusted conviction. The
  Analyst proposes; it never sizes or places trades (that is the Risk Engine, Phase 7).
