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
