"""Check the cost that Pi reports for Anthropic requests against verified
list prices, and correct wrong components. No routes.

Pi prices each request from its own model registry, and cost figures use
that reported cost (it is the only source that prices 1-hour cache writes,
which Pi rows do not split out). The registry was wrong for a while:

- Sonnet 5.5 cache reads at $0.20 per million after Anthropic cut them to
  $0.10 on 7 Oct 2026, until 8 to 12 Oct (the update date differs per
  machine); before the cut $0.20 was right;
- Fable 5.1 cache reads at $1.00 (the Fable 5 rate) or other values
  instead of $0.25 (its rate since its launch on 1 Sep 2026).

So input and cache-read components are recomputed from tokens at the
verified rate when they differ. Output anchors the check: a row whose
output component is not at the standard rate (fast mode, US-only
inference, a price Pi knows and we do not) is left alone. Cache writes
stay as reported (Pi requests 1-hour caching; its 2x-input rate is right
for every verified model). The original values are kept in
``events.reported_cost_pi_raw`` (JSON), so a later table change is applied
again from them.

Rates: platform.claude.com/docs/en/about-claude/pricing, read 2026-11-09.
Haiku 5.5 is not here: it has prompt-size tiers ($0.10/$0.50 up to 100k
prompt tokens, $0.50/$2.50 above), and Pi's tiered prices cannot be
checked against one rate.
"""

import json
from typing import Optional

from .pricing import SONNET55_CR_CUT_UTC, display_model

# Display name -> (input, output, cache read) per million tokens, or an
# ascending list of (effective_from_epoch, rates) as in pricing.MODEL_PRICING.
VERIFIED_RATES = {
    "Opus 5.5": (4.00, 20.00, 0.20),
    "Opus 5": (5.00, 25.00, 0.50),
    "Sonnet 5.5": [(0.0, (2.00, 10.00, 0.20)), (SONNET55_CR_CUT_UTC, (2.00, 10.00, 0.10))],
    "Sonnet 5": (2.00, 10.00, 0.20),
    "Fable 5.2": (8.00, 40.00, 0.20),
    "Fable 5.1": (10.00, 50.00, 0.25),
    "Fable 5": (10.00, 50.00, 1.00),
}
# Bump when VERIFIED_RATES or the rule changes: the next start re-checks
# every Pi Anthropic row.
VERSION = "3"  # 2: Sonnet 5.5 cache-read era; 3: Fable 5.2
VERSION_KEY = "pi_price_check_version"
OUTPUT_TOL = 0.02     # output within 2% of list: the standard tier
REL_TOL = 0.01        # a component off by more than 1% ...
ABS_TOL = 0.0001      # ... and more than $0.0001 is corrected
COMPONENTS = ("input", "output", "cache_read", "cache_write", "total")
_PAGE_ROWS = 20_000


def _off(got: float, want: float) -> bool:
    return abs(got - want) > max(REL_TOL * want, ABS_TOL)


def rates_at(model: Optional[str], ts: Optional[float]) -> Optional[tuple]:
    entry = VERIFIED_RATES.get(display_model(model)) if model else None
    if not isinstance(entry, list):
        return entry
    if ts is None:
        return None  # an era model needs the request time
    rates = entry[0][1]
    for since, r in entry:
        if since <= ts:
            rates = r
    return rates


def corrected(model: Optional[str], ts: Optional[float], inp: int, out: int, cr: int,
              reported: dict) -> Optional[dict]:
    """New reported components {input, output, cache_read, cache_write,
    total} for one Pi Anthropic row at request time ``ts``, or None when the
    row stays as is. ``reported`` holds the same keys (None where Pi sent
    nothing)."""
    rates = rates_at(model, ts)
    if rates is None or not out or reported.get("output") is None:
        return None
    rate_in, rate_out, rate_cr = rates
    if abs(reported["output"] - out * rate_out / 1e6) > OUTPUT_TOL * out * rate_out / 1e6:
        return None
    new = dict(reported)
    delta = 0.0
    for key, tokens, rate in (("input", inp, rate_in), ("cache_read", cr, rate_cr)):
        got = reported.get(key)
        if got is None or tokens is None:
            continue  # nothing to check against
        want = tokens * rate / 1e6
        if _off(got, want):
            new[key] = want
            delta += want - got
    if new == reported:
        return None
    if reported.get("total") is not None:
        new["total"] = reported["total"] + delta
    return new


def apply_to_row(row: dict) -> None:
    """Ingest: correct a new Pi row in place and keep the original values."""
    if (row.get("provider") or "").lower() != "anthropic":
        return
    reported = {k: row.get(f"reported_cost_{k}") for k in COMPONENTS}
    new = corrected(row.get("model"), row.get("ts_epoch"), row.get("input_tokens"),
                    row.get("output_tokens"), row.get("cache_read_tokens"), reported)
    if new is None:
        return
    row["reported_cost_pi_raw"] = json.dumps(reported)
    for k in COMPONENTS:
        row[f"reported_cost_{k}"] = new[k]


_SELECT = (
    "SELECT rowid, model, day, input_tokens, output_tokens, cache_read_tokens, "
    "reported_cost_input, reported_cost_output, reported_cost_cache_read, "
    "reported_cost_cache_write, reported_cost_total, reported_cost_pi_raw, ts_epoch "
    "FROM events WHERE source_client='pi-agent' AND LOWER(provider)='anthropic' "
    "AND type='assistant' AND rowid > ? ORDER BY rowid LIMIT ?")
_NEEDED = ("model", "day", "ts_epoch", "input_tokens", "output_tokens", "cache_read_tokens",
           "source_client", "provider", "type", "reported_cost_pi_raw",
           *(f"reported_cost_{k}" for k in COMPONENTS))
_UPDATE = (
    "UPDATE events SET reported_cost_input=?, reported_cost_output=?, "
    "reported_cost_cache_read=?, reported_cost_cache_write=?, reported_cost_total=?, "
    "reported_cost_pi_raw=? WHERE rowid=?")


def recheck_all(conn) -> set:
    """Startup, once per VERSION: re-check every Pi Anthropic row from its
    original values. Returns the days whose costs changed (their rollups
    must be rebuilt), with the days that a new price era in pricing.py
    reprices. The caller owns the transaction."""
    row = conn.execute("SELECT value FROM meta WHERE key=?", (VERSION_KEY,)).fetchone()
    if row is not None and row[0] == VERSION:
        return set()
    columns = {r[1] for r in conn.execute("PRAGMA table_info(events)")}
    if not columns.issuperset(_NEEDED):
        return set()  # a legacy table from before token capture holds no Pi rows
    days: set = set()
    last = 0
    while rows := conn.execute(_SELECT, (last, _PAGE_ROWS)).fetchall():
        updates = []
        for r in rows:
            current = dict(zip(COMPONENTS, r[6:11]))
            raw_json = r[11]
            original = json.loads(raw_json) if raw_json else current
            new = corrected(r[1], r[12], r[3], r[4], r[5], original)
            target = new if new is not None else original
            target_raw = json.dumps(original) if new is not None else None
            if target != current or target_raw != raw_json:
                updates.append(tuple(target[k] for k in COMPONENTS) + (target_raw, r[0]))
                days.add(r[2])
        if updates:
            conn.executemany(_UPDATE, updates)
        last = rows[-1][0]
    days |= _era_days(conn)
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (VERSION_KEY, VERSION))
    return {d for d in days if d}


def _era_days(conn) -> set:
    """Days with Sonnet 5.5 requests before its cache-read cut. Rows priced
    from tokens (SDK rows, Pi rows without a cost) were rolled up at the one
    undated LiteLLM price; pricing.MODEL_PRICING now has the era."""
    return {r[0] for r in conn.execute(
        "SELECT DISTINCT day FROM events WHERE type='assistant' "
        "AND model LIKE 'claude-sonnet-5-5%' AND ts_epoch < ?", (SONNET55_CR_CUT_UTC,))}
