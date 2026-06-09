# AI-Powered Sector-Specialist Trading Bot — Design

**Date:** 2026-06-09
**Status:** Approved design, pre-implementation
**Author:** mason@voltai.com (with Claude)

---

## 1. Purpose & Goals

Build an agentically-driven trading bot that deploys **real capital** by leveraging AI's
genuine strength — collecting, structuring, and interpreting messy unstructured data
(SEC filings, news, transcripts, sentiment) faster and more consistently than a human —
while keeping all execution and risk control in deterministic, testable code.

The bot **specializes in a single sector** (15–40 names) so its agents accumulate deep,
durable familiarity with the sector's drivers, players, and recurring patterns.

### Guiding principles
- **LLMs have no predictive price alpha.** Agents are used for *information processing
  and judgment*, never for execution or risk decisions.
- **The Risk Engine is a deterministic layer the agents cannot bypass.** Agents propose;
  code disposes.
- **Prove an edge on paper before risking a dollar.** Paper↔live is a single config flag
  on the same validated code path.
- **Trade where there is a domain edge** (sector chosen to match the operator's knowledge).
- **Specialization = a bounded, curated data universe**, not the whole market.

### Non-goals (YAGNI)
- No sub-minute / HFT trading (incompatible with LLM latency and cost).
- No fully-autonomous, unconstrained agent trading.
- No multi-sector coverage in v1.
- No vector/semantic memory store in v1 (interface leaves room for it later).

---

## 2. Key Decisions (from brainstorming)

| Dimension | Decision |
|---|---|
| Primary goal | Deploy real capital |
| Asset class | US equities first; options as a phase-2 expression layer |
| Time horizon | Swing (days–weeks) + hourly re-evaluation |
| Capital | Begin paper; scale to >$25k (clears PDT rule) once edge is proven |
| Specialization | One sector, 15–40 names + sector ETF benchmark |
| Autonomy | Auto-execute **within hard, code-enforced guardrails** |
| Data budget | Free/low-cost tiers first; upgrade only where backtests show measurable edge |
| Stack | Python, runs locally (designed to migrate to cloud for live) |
| Agent reasoning | Claude — tiered: Haiku 4.5 (triage), Sonnet 4.6 (deep read), Opus 4.8 (analyst/review) |
| Earnings calendar | Free source + manual override in `universe.yaml` |
| Memory retrieval | Keyed/structured (indexed by ticker & sector), behind a swappable interface |

### Model & cost reference (current Claude pricing, per 1M tokens)
- Opus 4.8 — $5 in / $25 out — analyst + review (hard reasoning, runs rarely).
- Sonnet 4.6 — $3 in / $15 out — deep read of material names.
- Haiku 4.5 — $1 in / $5 out — per-name "is anything new?" triage.

Estimated **$0.05–$0.40 per full agent cycle** with correct tiering + prompt caching;
a few dollars/day at hourly cadence — negligible vs. capital.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (Python)                       │
│         scheduler · runs the cycle hourly + EOD · run journal  │
└───────┬─────────────────────────────────────────────┬─────────┘
        │                                               │
        ▼                                               ▼
┌───────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────┐
│ DATA LAYER     │   │ RESEARCH     │   │ ANALYST      │   │ RISK ENGINE│
│ pluggable      │──▶│ AGENTS       │──▶│ AGENT        │──▶│ (pure code)│
│ providers      │   │ (Claude)     │   │ (Claude)     │   │ HARD limits│
│ prices/filings │   │ news/filings │   │ synthesize   │   │ NO LLM     │
│ /news (cached) │   │ → structured │   │ thesis →     │   │ veto power │
└───────────────┘   │   signals    │   │ trade plan   │   └─────┬──────┘
        ▲           └──────────────┘   └──────────────┘         │
        │                                                        ▼
┌───────────────┐                                        ┌────────────┐
│ MEMORY/STATE   │◀───── agents read & write ────────────│ EXECUTION  │
│ dossiers·journal│       sector knowledge accrues here    │ ADAPTER    │
│ ·lessons·P&L   │                                        │ paper│live │
└───────────────┘                                        └────────────┘
```

Eight components, each with one job and a well-defined interface. The orchestrator is a
plain Python scheduler (not an LLM) that drives a fixed, auditable pipeline each cycle:
**Data → Research → Analyst → Risk → Execution**, with Memory underpinning all of it.

---

## 4. Components

### 4.1 Orchestrator
- Deterministic Python scheduler. Two cadences: **hourly** (intraday re-evaluation) and
  **EOD** (full review + maintenance).
- Drives the fixed pipeline; owns control flow, retries, and error handling.
- Maintains a **run journal** (every cycle's actions, errors, regime notes) — queryable
  and summarized into agent context so the system is aware of its own operating history.
- Honors the global **kill switch** before every cycle.

### 4.2 Data Layer
- **Pluggable provider protocol** so free→paid swaps are config changes:
  - `PriceProvider.get_bars(ticker, timeframe, lookback) -> DataFrame`
  - `FilingsProvider.get_recent_filings(ticker, types, since) -> list[Filing]`
  - `NewsProvider.get_news(ticker | sector, since) -> list[NewsItem]`
  - `FundamentalsProvider.get_fundamentals(ticker) -> Fundamentals`
- **Phase-1 free implementations:** prices via broker API (Alpaca/IBKR) + yfinance
  fallback; filings via SEC EDGAR; news via free RSS/headlines; fundamentals via
  yfinance/FMP free tier.
- **Caching (cost-critical):** SQLite + parquet on disk in front of every provider.
  Filings/fundamentals cached for days; the **agent's structured extraction of a filing
  is cached by filing hash** so an unchanged 10-K is never re-sent to Claude. News
  deduped by URL hash. Only newest price bars fetched per cycle.
- **Sector universe is config:** `universe.yaml` defines tickers, the benchmark ETF, and
  per-ticker metadata (earnings dates, market-cap tier). Data is only ever pulled for
  names in this file.
- **Earnings calendar:** free source auto-populated, **manually overridable per ticker**
  in `universe.yaml`. The layer warns when a position approaches an unconfirmed date.
- **Point-in-time capable:** can serve data "as of date T" with no lookahead — required
  for honest backtesting.
- **Normalization:** all providers return the same typed objects regardless of source.

### 4.3 Research Agents
- **Job:** read new/changed data for a ticker (filings, news, transcripts) and emit a
  structured signal. Parallelizable — one per ticker/cluster.
- **Model tiering (the cost lever):** a **Haiku 4.5** triage pass decides "is there
  anything new/material here?" per name. Only names that pass get a **Sonnet 4.6** deep
  read (Opus for the few that look pivotal). Most names on most hours touch no expensive
  model.
- **Structured output (validated JSON):**
  ```
  { ticker, signal_type: bullish|bearish|neutral,
    conviction: 0-1, time_horizon, catalyst,
    evidence: [{source, quote, filing_url}],   // citations REQUIRED
    risk_flags: [...] }
  ```
- **Mandatory citations** — a signal whose claims have no source quote/URL is dropped by
  the orchestrator before it reaches the Analyst. This is the primary hallucination guard.

### 4.4 Analyst Agent
- **Job:** synthesize all research signals + price/technical context + current positions +
  the sector thesis from memory into a ranked **trade plan**.
- **Model:** **Opus 4.8**, `effort: high`, adaptive thinking. Runs once per cycle — the
  one genuinely hard reasoning step; cost is trivial, quality matters most.
- **Structured output (validated JSON):**
  ```
  { proposals: [
      { ticker, action: buy|sell|hold|trim|add,
        target_weight, conviction, thesis,
        entry_logic, exit_logic, stop_loss,
        supporting_signal_ids: [...] } ],
    sector_view, portfolio_notes }
  ```
- Produces **proposals, not orders** — recommends target weights only; never sizes or
  places trades. Because it sees the whole basket + benchmark, it can propose
  **relative-value** views (long the leader, trim the laggard).

### 4.5 Risk Engine
- **Pure Python. No LLM. No exceptions.** Receives proposals; approves, downsizes, or
  vetoes each against hard-coded rules:
  - **Position sizing** — max % per name; volatility-scaled.
  - **Sector/portfolio caps** — max gross/net exposure, max correlated-cluster exposure.
  - **Daily loss circuit breaker** — halt new entries (optionally flatten) on day-P&L floor.
  - **Earnings blackout** — no new position within N days of a confirmed earnings date.
  - **Liquidity/sanity** — min volume, max spread, no fat-finger sizes, no orders during halts.
  - **Kill switch** — single flag (file/env) stops all trading immediately.
- Every decision logged **with the rule that fired.** Analyst conviction influences sizing
  *within* limits but can never exceed them.

### 4.6 Execution Adapter
- One interface; **paper or live by a single config flag**, so validated code is the code
  that trades. Phase-1 broker: Alpaca or IBKR.
- Handles order placement, fill tracking, and reconciliation against the position store.

### 4.7 Autodidactic Memory & Awareness System
Three layers of memory + a learning loop + disciplined context injection.

- **Layer 1 — Semantic memory (evolving understanding):** versioned Markdown/JSON
  dossiers the agents read and extend.
  - *Sector dossier:* key players, demand drivers, supply chain, what moves the group,
    cross-correlations.
  - *Per-ticker dossiers:* business model, revenue drivers, "what actually moves this
    stock," recurring patterns. Effectively a `CLAUDE.md` the bot maintains about its
    own domain.
- **Layer 2 — Episodic memory (trade journal):** append-only ground truth. Every cycle's
  proposals, Risk Engine verdicts, fills, attached reasoning/evidence, outcome, and P&L.
- **Layer 3 — Distilled lessons (calibration & patterns):** the autodidactic output.
  ```
  { pattern, evidence_trades:[...], lesson, confidence,
    applies_to: ticker|sector, supersedes: lesson_id? }
  ```
  Plus **calibration tracking** — when conviction was 0.8, was it right ~80%? Systematic
  over/under-confidence is surfaced and fed back.

- **Learning loop:**
  ```
  trade closes → Review Agent (Opus 4.8) compares predicted vs actual
     → writes/updates a distilled lesson + updates calibration stats
     → lesson promoted into the relevant dossier (semantic memory)
     → next cycle, that lesson is injected into Analyst + Research context
  ```
  Over weeks, the dossiers become the bot's own *earned* understanding, not just seed text.

- **Awareness — how memory reaches agents each cycle (and stays cheap):**
  - Each agent's prompt is **assembled from memory**: Analyst gets sector dossier +
    dossiers for names in play + recent relevant lessons + open positions + its own
    calibration record.
  - **Prompt caching** keeps the stable knowledge prefix (sector dossier, system prompt,
    tool defs) cached across cycles (~0.1× on reads); volatile data goes after the cache
    breakpoint.
  - **Keyed/structured retrieval** behind a `MemoryRetriever` interface — loads dossiers/
    lessons for exactly the tickers in the current shortlist. Deterministic and
    debuggable; a vector backend can be added later without touching agents.
  - The orchestrator's run journal is summarized into context so the system knows its own
    history, not just market data.

### 4.8 Validation, Backtesting & Go-Live Gating
- **Backtesting harness** — replays historical data through the *same* Analyst/Risk/
  Execution code path (agents run against point-in-time data, no lookahead). Also used to
  tune Risk Engine parameters.
- **Point-in-time discipline** — the data layer forbids lookahead by construction.
- **Paper-trading phase (months)** — live code path, execution adapter set to paper.
  Tracks walk-forward performance, **calibration**, and **agent quality** (do cited
  signals hold up?).
- **Explicit go-live gate** — written checklist that must be met before flipping to live:
  minimum track-record length, positive risk-adjusted return net of fees, calibration
  within tolerance, max drawdown under limit, zero Risk Engine violations.
- **Observability** — structured per-cycle logging, daily summary (positions, P&L, agent
  reasoning), alerting on circuit-breaker trips and agent/data failures.

---

## 5. Data Flow (one hourly cycle)

1. Orchestrator checks kill switch and market hours; starts cycle, opens journal entry.
2. Data Layer refreshes prices/news/filings for the universe (cache-first, point-in-time).
3. Research Agents: Haiku triage across all names → Sonnet/Opus deep read on names with
   new material → structured, cited signals. Uncited signals dropped.
4. Analyst Agent (Opus): memory-assembled context + signals + positions → ranked proposals.
5. Risk Engine: approve/downsize/veto each proposal against hard limits; log rule firings.
6. Execution Adapter: place approved orders (paper or live); record fills.
7. Memory: append episodic record; on closed positions, Review Agent writes lessons +
   updates calibration + promotes lessons into dossiers.
8. Orchestrator: close journal entry; emit daily summary at EOD.

---

## 6. Build Sequence (high level — detailed plan to follow)

1. Project scaffold, config (`universe.yaml`, settings), typed data models.
2. Data Layer + caching + point-in-time, with free providers and one broker (paper).
3. Memory stores (semantic/episodic/lessons) + `MemoryRetriever` interface.
4. Research Agents with model tiering + structured outputs + citation enforcement.
5. Analyst Agent + structured trade-plan output.
6. Risk Engine + kill switch + full rule logging.
7. Execution Adapter (paper) + position/fill reconciliation.
8. Orchestrator wiring + run journal + observability/alerting.
9. Backtesting harness over the same code path.
10. Review Agent + learning loop + calibration tracking.
11. Paper-trading operation; iterate; evaluate against the go-live gate.

---

## 7. Open Items / Future
- Sector selection finalized in `universe.yaml` (lean toward the operator's domain edge).
- Options expression layer (phase 2) — greeks/vol modeling added only after equities proven.
- Paid data feeds — adopt only where backtests show measurable, specific edge.
- Vector/semantic memory — add behind the existing retriever interface if/when the lesson
  corpus is large enough to need fuzzy recall.
- Cloud deployment for 24/7 live operation (migrate from local after go-live).

---

## 8. Risks & Mitigations
- **No edge exists** → paper-first + explicit go-live gate; refuse to deploy without proof.
- **Agent hallucination** → mandatory citations, structured outputs, Risk Engine veto.
- **Backtest lookahead** → point-in-time data interface enforced by construction.
- **Runaway loss** → daily circuit breaker, position/exposure caps, kill switch.
- **Token cost blowup** → model tiering, prompt caching, cached filing extractions.
- **Stale earnings data** → manual override calendar + proximity warnings.
- **PDT rule** → swing-only until capital >$25k; configurable day-trade guard.
