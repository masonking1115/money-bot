# Catalyst-Driven Long Strategy — Design

**Date:** 2026-06-09
**Status:** Approved design, pre-implementation (informs Plans 4–5)
**Author:** mason@voltai.com (with Claude)
**Parent spec:** `2026-06-09-ai-trading-bot-design.md`

---

## 1. Purpose

Define the concrete, testable trading strategy the bot will run, so the Research agents
(Plan 4) and the Analyst agent (Plan 5) have a precise specification to implement. This is
the **first** strategy; the architecture stays strategy-agnostic so others can be added and
backtested later.

The operator works in the **semiconductor industry** (genuine domain edge) but has limited
trading experience. The design therefore (a) leverages that domain edge — both by seeding the
bot's knowledge and by using agents to read catalysts — and (b) keeps all *trading* discipline
mechanical and system-enforced, with parameters tuned by backtesting rather than set by feel.

## 2. Strategy Summary

- **Name:** Catalyst-Driven Long
- **Sector:** Semiconductors / AI hardware (universe of 15–40 names + SMH benchmark)
- **Posture:** Long-only, with an *optional* SMH (sector ETF) hedge. No single-name shorting.
- **Horizon:** Swing (days–weeks), re-evaluated hourly.
- **Thesis:** Fresh, material catalysts in semis (guidance changes, demand/supply signals,
  policy shifts) produce multi-day price *drift* because the market digests dense, technical
  information slowly. Agents that read that information carefully and consistently can enter
  the strongest catalyst names early and exit before the drift decays.

> *Drift* = the tendency of a stock to keep moving in the catalyst's direction for several
> days after the news, rather than repricing instantly.

## 2.5 Strategy as a Pluggable Module (projectized)

Strategies are first-class, swappable plugins so the operator can experiment with others later
without touching the agent/orchestrator plumbing. Catalyst-Driven Long is simply the **first**
plugin.

A `Strategy` is a self-contained module under `moneybot/strategies/` exposing a common interface:
```
class Strategy(Protocol):
    name: str
    def signal_schema(self) -> dict              # JSON schema for the research signal it wants
    def research_guidance(self) -> str           # prompt fragment: what catalysts to look for
    def rank(self, signals, context) -> list[Proposal]   # entry selection/ranking
    def exit_plan(self) -> ExitPlan              # exit rules + thesis-check guidance
    def parameters(self) -> StrategyParams       # tunable defaults
```
A `StrategyRegistry` registers strategies by name; the active one is chosen by config
(`MONEYBOT_STRATEGY=catalyst_driven`, default). The Research and Analyst agents are **generic
executors**: they read the active strategy's `signal_schema`/`research_guidance` to know what to
extract, and call its `rank`/`exit_plan` for decisions. Adding a new strategy = add a module +
register it; no changes to agents, Risk Engine, orchestrator, or backtester. The backtester can
run any registered strategy, enabling head-to-head comparison.

The rest of this document specifies the **Catalyst-Driven Long** plugin concretely.

## 3. Catalyst Taxonomy (the Research signal)

A Research agent reads the recent filings/news for each name and emits a structured,
**citation-required** signal. Categories the agent classifies:

- **Earnings / guidance** — a beat *with raised forward guidance* (guidance weighted more
  heavily than the reported quarter — the dominant driver in semis).
- **Demand** — hyperscaler capex commentary, design wins, large bookings/backlog.
- **Supply** — capacity tightening/loosening, foundry/node news, inventory normalization.
- **Policy / macro** — export-control changes, tariffs, subsidies (CHIPS-type).

Each emitted catalyst signal carries:
```
{ ticker, category, direction,           # only "bullish" is acted on (long-only)
  materiality: 0-1, freshness_days: int,
  conviction: 0-1,
  evidence: [{source, quote, url}],       # MANDATORY — no citation → signal dropped
  thesis: str }
```

## 4. Entry Logic (Analyst, once per cycle)

1. Collect bullish catalyst signals for the universe.
2. **Freshness gate:** a catalyst must be within the freshness window (default ~5 trading
   days) to open a *new* position — stale catalysts are already priced in.
3. **Rank** candidates by `materiality × conviction × freshness_decay`, with a
   relative-strength tiebreaker (prefer names already firm vs SMH — don't fight the tape).
4. The Analyst independently confirms each top-ranked thesis, then proposes longs up to the
   portfolio's open-slot and exposure limits.
5. Each proposal includes: the thesis, the catalyst it rests on (signal id), conviction, and
   the exit plan (Section 5). Proposals are recommendations — the Risk Engine sizes/approves.

## 5. Exit Logic (hybrid — first trigger wins)

A position closes when **any** of these fires:
- **Thesis invalidation** — each cycle the agent re-reads the name; if the catalyst broke
  (guidance walked back, design win lost, etc.) it exits. Uses the domain edge on the exit side.
- **Time stop** — max-hold timer (default ~10 trading days) since drift decays.
- **Stop-loss** — hard automatic sell if price falls a set % below entry (caps per-trade loss).
- **Profit-target** — automatic sell after a set % gain (locks the win).

Stop-loss and profit-target are mechanical and enforced by the Risk Engine regardless of agent
state. ("Trading days" = market-open days.)

## 6. Position Sizing & Hedge

- **Conviction-weighted within hard caps:** higher-conviction catalysts get a larger slice,
  bounded by a max % per name and a max total-invested cap.
- **Volatility-scaling:** a more volatile name gets a smaller position than a steady one for
  the same conviction, so no single name dominates portfolio risk.
- **Optional SMH hedge:** holds a small offsetting position against the semis ETF to neutralize
  *sector beta* (how much a name moves with the whole sector), so returns come from name
  selection rather than the sector rising. Built in but switchable; backtesting decides if it
  helps. Sizing/hedge are owned by the Risk Engine (Plan 6), not the agents.

## 7. Parameters (defaults; all backtest-tuned)

| Parameter | Default | Meaning |
|---|---|---|
| `freshness_window_days` | 5 | Max catalyst age to open a new position |
| `max_hold_days` | 10 | Time stop |
| `stop_loss_pct` | 8% | Hard per-trade loss cap |
| `profit_target_pct` | 20% | Win lock-in |
| `max_position_pct` | configurable | Max % of capital per name |
| `max_sector_exposure_pct` | configurable | Cap on total long exposure |
| `hedge_enabled` | false (phase 1) | Toggle the SMH hedge |

Defaults are starting points only; the backtest (Plan 9) and paper phase tune them to the
operator's risk tolerance before any real capital is deployed.

## 8. Operator Setup: Seed the Sector Dossier

Before live/paper operation, the operator seeds the **sector dossier** (semantic memory,
`SemanticStore` key `sector:semiconductors`) and per-ticker dossiers with their industry
knowledge: what actually moves each name, which supply-chain signals are real tells, key
customers/suppliers, and what "good guidance" looks like per company. The Research and Analyst
agents read this every cycle, so the bot starts with the operator's edge and refines it from
outcomes (the learning loop, Plan 10).

## 9. How This Maps to the Build

A **Strategy Framework** plan now precedes the agents (the `Strategy` interface + registry +
the Catalyst-Driven Long plugin from Sections 2.5–7). The roadmap shifts by one: Strategy
Framework → Research agents → Analyst → Risk Engine → Execution → Orchestrator → Backtest →
Learning loop.

- **Strategy Framework (next plan):** `Strategy` protocol, `StrategyRegistry`, config selection,
  and the `CatalystDrivenLong` plugin encoding Sections 3–7 (signal schema, research guidance,
  ranking, exit plan, parameters). Pure logic — independently unit-testable, no agents/LLM yet.
- **Research agents:** generic tiered agents that read the active strategy's `signal_schema` +
  `research_guidance` and emit the structured, citation-required signal (Section 3).
- **Analyst:** generic agent that calls the active strategy's `rank` + freshness gate and
  independently confirms theses, emitting proposals carrying the exit plan (Sections 4–5).
- **Risk Engine:** owns sizing/volatility-scaling/exposure caps and enforces stop-loss,
  profit-target, time-stop, and the optional hedge (Sections 5–6).
- **Backtest:** runs *any* registered strategy to tune Section 7 parameters and compare
  strategies head-to-head on history.
- **Learning loop:** the Review Agent writes lessons (e.g. "this name's beats are priced in")
  back into the dossiers, sharpening future catalyst scoring.

## 10. Risks & Honest Caveats

- **The edge may not survive costs/noise.** Catalyst drift is documented but modest; the
  paper phase + go-live gate must prove a real, fee-net edge before real capital.
- **Catalyst staleness / data latency.** Free news/filings can lag; the freshness window and
  point-in-time discipline mitigate acting on already-priced information.
- **Agent misjudgment of materiality.** Mandatory citations + the Risk Engine's hard caps
  bound the damage from a wrong call; the learning loop calibrates over time.
- **Sector-wide drawdowns.** Long-only is exposed to a sector sell-off; exposure caps and the
  optional hedge are the mitigations.
- **Single strategy.** Starting with one strategy concentrates risk; additional candidates can
  be added and backtested once this one is validated.
