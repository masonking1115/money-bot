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

Phase 1 (this plan): foundation + point-in-time price data layer.
