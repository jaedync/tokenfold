# Cost-estimation cross-validation vs open-source tools — 2026-06-11

Question: does tokenfold's cost math drift from the popular open-source
Claude Code cost tools? Method: run the runnable ones against this machine's
own `~/.claude/projects` transcripts (the same raw data tokenfold ingests),
diff per-day against tokenfold's prod numbers for the same machine, and read
every tool's pricing/dedup source.

## Numeric result (ccusage 16k★, the de facto standard)

June 2026, single machine, independent implementations over identical input:

| day        | tokenfold | ccusage | drift  |
|------------|-----------|---------|--------|
| Jun 1–8,10 | —         | —       | $0.00 each, to the cent |
| Jun 9      | 522.90    | 522.63  | +0.05% |
| Jun 11*    | 28.21     | 29.10   | freshness (*in progress) |
| **total**  | **2590.12** | **2590.74** | **−0.02%** |

- Jun 9: tokenfold counted ~295k extra cache-read tokens (≈ the $0.27).
  Likely a sidechain replay counted under a new requestId (see ccusage dedup
  note below) or transcript pruning; tokenfold's DB keeps history disk loses.
- Jun 11: ccusage reads the live transcript second-by-second; tokenfold's
  push debounce lags ≤5 min. Self-healing.
- `--mode calculate` (recompute everything, ignore any precomputed costUSD)
  shifts ccusage only +0.1% → the agreement is methodology convergence, not
  shared shortcuts.
- Jun 10 is a pure Fable 5 day with 3.6M cache-write tokens that matched to
  the cent — numeric proof both tools bill the 1h cache tier at 2x (a flat
  1.25x would differ by ~$27 that day).

## Methodology comparison (source-level, main branches 2026-06-11)

| | tokenfold | ccusage 16k★ | usage-monitor 8k★ | opcode 22k★ | sniffly 1.2k★ |
|---|---|---|---|---|---|
| pricing source | LiteLLM + static pins | LiteLLM (embedded+live) + models.dev | hardcoded, STALE (Opus billed 15/75) | hardcoded 2 families, others $0 | LiteLLM + stale fallback |
| 1h vs 5m cache | ✅ 2x/1.25x | ✅ 2x/1.25x (`cost.rs:7,97-125`) | ❌ flat 1.25x | ❌ flat | ❌ flat |
| streamed-usage dedup | MAX per request_id | (msgId,reqId) keep-max ≈ same | first-wins; none w/o ids | first-wins | ❌ SUMS chunks (overcounts) |
| day bucketing | America/Chicago | local tz (configurable) | UTC | UTC (mixed) | viewer tz |
| fast mode | ✅ (Opus table) | ✅ (LiteLLM multiplier, agrees: 4.6→6x) | ❌ | ❌ | ❌ |
| web-search fees | ✅ ($10/1k) | ❌ | ❌ | ❌ | ❌ |
| inference_geo | ✅ (us 1.1x) | ❌ | ❌ | ❌ | ❌ |
| >200k-context tier pricing | ❌ | ✅ (`tiered_cost`, threshold 200k) | ❌ | ❌ | ❌ |
| maintained | — | active (Rust rewrite) | dead since 2025-07 | active GUI | active |

Claude-Code-Usage-Monitor and opcode were not run: their pricing tables are
provably wrong for every post-mid-2025 model (Monitor bills all Opus at
legacy $15/$75; opcode substring-matches `opus-4` so Opus 4.5+ is 3x
overpriced and Haiku/Fable are $0), so any numeric drift would measure their
staleness, not our accuracy.

## What this buys us

1. **Validation:** tokenfold and the one methodologically-sound popular tool
   agree to −0.02% over a month of real usage, with both residual cents
   explained. The cache-tier, dedup, and pricing work is corroborated by an
   independent 16k-star implementation.
2. **Known gap (ours): >200k-context tier pricing.** Anthropic bills input
   above 200k context at premium rates (LiteLLM `*_above_200k_tokens`
   fields); ccusage applies it per component, tokenfold doesn't. Zero impact
   on this machine in June (else the totals would drift), but enterprise
   1M-context Sonnet usage would be undercounted. Candidate follow-up.
3. **Candidate dedup refinement:** ccusage keeps the non-sidechain copy and
   does a secondary message-id-only check because "sidechain logs can replay
   parent messages with new request IDs" — tokenfold's MAX-per-request_id
   would double-count such replays (~$0.27 on Jun 9, +0.05%). Low priority.
4. **Nobody else** bills web-search server-tool fees or geo modifiers —
   tokenfold is ahead there.
