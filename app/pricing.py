"""Model pricing: static rates + LiteLLM dynamic fetch with caching."""

import json
import time
import urllib.request
from datetime import datetime, timezone

from .config import LITELLM_URL, PRICING_CACHE_TTL
from .db import get_conn, write_txn

MODEL_DISPLAY = {
    "claude-fable-5": "Fable 5",
    "claude-opus-4-8": "Opus 4.8",
    "claude-opus-4-7": "Opus 4.7",
    "claude-opus-4-6": "Opus 4.6",
    "claude-opus-4-5-20251101": "Opus 4.5",
    "claude-sonnet-5": "Sonnet 5",
    "claude-sonnet-4-6": "Sonnet 4.6",
    "claude-sonnet-4-5-20250929": "Sonnet 4.5",
    "claude-sonnet-4-20250514": "Sonnet 4",
    "claude-haiku-4-5-20251001": "Haiku 4.5",
    "claude-3-5-sonnet-20241022": "Sonnet 3.5",
    "claude-3-5-haiku-20241022": "Haiku 3.5",
}

# Sonnet 5 intro pricing ends here; billing cutover timezone assumed UTC.
SEPT1_2026_UTC = datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp()

# Pricing per MTok: (input, output, cache_write_5m, cache_read).
# Values are either a constant 4-tuple, or an ASCENDING list of
# (effective_from_epoch_utc, 4-tuple) eras selected by EVENT timestamp —
# never wall clock — so re-summarizing history can't reprice old usage.
MODEL_PRICING = {
    "Fable 5":   (10.00, 50.00, 12.50, 1.00),
    "Opus 4.8":  (5.00, 25.00, 6.25, 0.50),
    "Opus 4.7":  (5.00, 25.00, 6.25, 0.50),
    "Opus 4.6":  (5.00, 25.00, 6.25, 0.50),
    "Opus 4.5":  (5.00, 25.00, 6.25, 0.50),
    # Sonnet 5: intro rates through August 2026, standard from Sept 1 UTC.
    # Standard-period cache rates 3.75/0.30 are ASSUMED 1.25x/0.1x of the $3
    # base pending Anthropic confirmation (only intro cache rates published).
    "Sonnet 5": [
        (0.0, (2.00, 10.00, 2.50, 0.20)),
        (SEPT1_2026_UTC, (3.00, 15.00, 3.75, 0.30)),
    ],
    "Sonnet 4.6": (3.00, 15.00, 3.75, 0.30),
    "Sonnet 4.5": (3.00, 15.00, 3.75, 0.30),
    "Sonnet 4":   (3.00, 15.00, 3.75, 0.30),
    "Sonnet 3.5": (3.00, 15.00, 3.75, 0.30),
    "Haiku 4.5":  (1.00, 5.00, 1.25, 0.10),
    "Haiku 3.5":  (0.80, 4.00, 1.00, 0.08),
}
# No silent fallback: a model without confirmed pricing contributes $0 to every
# cost figure and is flagged via is_priced() so the UI shows an em-dash instead
# of a fabricated number. (The old behavior silently billed unknown models at
# Sonnet rates — Fable 5 launched at 2x Opus and was undercounted ~3-4x.)
_UNPRICED = (0.0, 0.0, 0.0, 0.0)

# Encountering an unknown model forces a TTL-bypassing LiteLLM refresh so new
# models get real pricing within minutes of first appearing, not up to 24h
# later. Rate-limited so a permanently-unknown name can't hammer GitHub.
UNKNOWN_REFRESH_INTERVAL_S = 900
_unknown_refresh_ts = 0.0

# Fast-mode Opus base (input, output) per MTok; cache rates re-derived from base.
# Keyed by display name — only Opus has a fast tier.
FAST_OPUS_BASE = {
    "Opus 4.8": (10.0, 50.0),
    "Opus 4.7": (30.0, 150.0),
    "Opus 4.6": (30.0, 150.0),
}
GEO_US_MULT = 1.1


def effective_geo(inference_geo, *, enterprise: bool):
    """The geo to bill at. Transcripts stamp 'not_available' on subscription
    traffic, so a US-pinned enterprise workspace is invisible — when the
    instance assumes US residency (config.ENTERPRISE_ASSUME_GEO == 'us'),
    enterprise usage bills at the US rate regardless of the recorded value.
    Reads config at call time so the assumption is flip-able without restarts
    in tests (prod flips via env + container restart + day re-roll)."""
    import app.config as config
    if enterprise and config.ENTERPRISE_ASSUME_GEO == "us":
        return "us"
    return inference_geo

# Server tools: web search bills a flat $10 per 1,000 requests ON TOP of token
# cost (web fetch is free — its fetched-content tokens are already in usage).
# Model-independent, and per-request fees take no geo/fast multipliers.
WEB_SEARCH_PER_1K = 10.0

# Canonical model sort order: Opus > Sonnet > Haiku, then version descending
MODEL_ORDER = list(MODEL_PRICING.keys())


def model_sort_key(name: str) -> int:
    try:
        return MODEL_ORDER.index(name)
    except ValueError:
        return len(MODEL_ORDER)  # unknown models sort last

# No "Sonnet 5" entry: no Anthropic-published scores on file — never fabricate.
MODEL_BENCHMARKS = {
    "Opus 4.6":   {"SWE-bench": 80.8, "Terminal-Bench": 65.4, "OSWorld": 72.7, "ARC-AGI-2": 68.8},
    "Opus 4.5":   {"SWE-bench": 80.9, "Terminal-Bench": 59.8, "OSWorld": 66.3, "ARC-AGI-2": 37.6},
    "Sonnet 4.6": {"SWE-bench": 79.6, "Terminal-Bench": 59.1, "OSWorld": 72.5, "ARC-AGI-2": 58.3},
    "Sonnet 4.5": {"SWE-bench": 77.2, "Terminal-Bench": 51.0, "OSWorld": 61.4, "ARC-AGI-2": 13.6},
    "Haiku 4.5":  {"SWE-bench": 73.3, "Terminal-Bench": 40.2, "OSWorld": 50.7},
}

_dynamic_pricing: dict = {}


def display_model(mid: str) -> str:
    if mid in MODEL_DISPLAY:
        return MODEL_DISPLAY[mid]
    name = mid
    if name.startswith("claude-"):
        name = name[7:]
    parts = name.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) >= 8 and parts[1][:8].isdigit():
        name = parts[0]
    segs = name.rsplit("-", 2)
    if (len(segs) >= 3 and segs[-1].isdigit() and len(segs[-1]) == 1
            and segs[-2].isdigit() and len(segs[-2]) == 1):
        base = "-".join(segs[:-2])
        return f"{base.replace('-', ' ').title()} {segs[-2]}.{segs[-1]}"
    return name.replace("-", " ").title()


def load_pricing(force=False):
    """Fetch Claude pricing from LiteLLM GitHub, with 24h DB-backed cache.
    force=True bypasses a still-fresh cache (used when an unknown model shows
    up — the cache may simply predate the model's addition to LiteLLM)."""
    global _dynamic_pricing

    conn = get_conn()
    # Check DB cache
    row = conn.execute("SELECT value FROM meta WHERE key='pricing_cache'").fetchone()
    if row and not force:
        try:
            cache = json.loads(row["value"])
            if time.time() - cache.get("ts", 0) < PRICING_CACHE_TTL:
                _dynamic_pricing = cache["pricing"]
                return
        except (json.JSONDecodeError, KeyError):
            pass

    # Fetch from LiteLLM
    fetched = None
    try:
        req = urllib.request.Request(LITELLM_URL, headers={"User-Agent": "tokenfold/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            fetched = json.loads(resp.read())
    except Exception:
        pass

    if fetched:
        pricing = {}
        for key, info in fetched.items():
            if not key.startswith("claude-") or "/" in key:
                continue
            inp = info.get("input_cost_per_token")
            out = info.get("output_cost_per_token")
            if inp is None or out is None:
                continue
            cw = info.get("cache_creation_input_token_cost", inp * 1.25)
            cr = info.get("cache_read_input_token_cost", inp * 0.1)
            dname = display_model(key)
            pricing[dname] = (
                round(inp * 1e6, 4),
                round(out * 1e6, 4),
                round(cw * 1e6, 4),
                round(cr * 1e6, 4),
            )
        if pricing:
            _dynamic_pricing = pricing
            try:
                with write_txn(conn) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                        ("pricing_cache", json.dumps({"ts": time.time(), "pricing": pricing})),
                    )
            except Exception:
                pass
            return

    # Fallback: use stale cache
    if row:
        try:
            cache = json.loads(row["value"])
            _dynamic_pricing = cache.get("pricing", {})
        except (json.JSONDecodeError, KeyError):
            pass


def is_priced(model_name: str) -> bool:
    """True when the model has CONFIRMED pricing (LiteLLM-fetched or static)."""
    return model_name in _dynamic_pricing or model_name in MODEL_PRICING


def _maybe_refresh_for_unknown(model_name: str):
    """Force a TTL-bypassing pricing refresh for an unseen model, rate-limited."""
    global _unknown_refresh_ts
    now = time.time()
    if now - _unknown_refresh_ts < UNKNOWN_REFRESH_INTERVAL_S:
        return
    _unknown_refresh_ts = now
    print(f"[pricing] unknown model {model_name!r} — forcing LiteLLM refresh")
    try:
        load_pricing(force=True)
    except Exception as e:
        print(f"[pricing] forced refresh failed: {e}")


def _resolve_eras(era_list: list, ts_epoch: float) -> tuple:
    """Select the last era with effective_from <= ts_epoch (list is ascending)."""
    rates = era_list[0][1]
    for eff, r in era_list:
        if eff <= ts_epoch:
            rates = r
    return rates


def era_boundaries() -> list[float]:
    """Every effective_from > 0 across era-listed models, sorted ascending.
    cost_windows splits its SQL aggregation groups at these edges."""
    bounds = {eff for val in MODEL_PRICING.values() if isinstance(val, list)
              for eff, _ in val if eff > 0}
    return sorted(bounds)


def get_pricing(model_name: str, ts_epoch: float | None = None) -> tuple:
    static = MODEL_PRICING.get(model_name)
    if isinstance(static, list):
        # Era-listed models: static wins over LiteLLM, which carries a single
        # undated price that is guaranteed wrong for one era (the live
        # pricing_cache is already poisoned with standard-rate Sonnet 5).
        return _resolve_eras(static, time.time() if ts_epoch is None else ts_epoch)
    if model_name in _dynamic_pricing:
        return _dynamic_pricing[model_name]
    if static is not None:
        return static
    # Unknown: maybe LiteLLM knows it and our cache is just older than the
    # model — refresh once, then re-check. Otherwise it is unpriced ($0).
    _maybe_refresh_for_unknown(model_name)
    if model_name in _dynamic_pricing:
        return _dynamic_pricing[model_name]
    return _UNPRICED


def compute_cost(model_name: str, inp: int, out: int, cw: int, cr: int,
                 speed: str | None = None, inference_geo: str | None = None,
                 cw_5m: int = 0, cw_1h: int = 0, web_search: int = 0,
                 ts_epoch: float | None = None) -> float:
    """List-price cost. Optional speed='fast' (Opus only) and inference_geo='us'
    modifiers layer on the LiteLLM-or-static base rate; defaults reproduce prior
    pricing exactly. Mirrors claude-usage-telemetry's effective_rates().

    ts_epoch selects the pricing era for date-effective models (None = now);
    the 1h-cache premium and fast/geo multipliers derive from the era-resolved
    base, so every modifier tracks the era.

    cw is the TOTAL cache-write tokens; cw_5m/cw_1h are the ephemeral-duration
    split when known. 5m writes bill at 1.25x base input (the cw_p rate), 1h
    writes at 2x base input. Any unsplit remainder — and all legacy rows that
    predate split capture — bills at the 5m rate (the historical behavior).
    A split that exceeds cw is untrusted transcript input and is ignored.

    web_search is the request COUNT (not tokens): flat WEB_SEARCH_PER_1K fee,
    charged even for unpriced models — the fee is confirmed, model-independent
    pricing, unlike token rates which are $0 when unconfirmed."""
    base_in, out_p, cw_p, cr_p = get_pricing(model_name, ts_epoch)
    if (speed or "").lower() == "fast" and model_name in FAST_OPUS_BASE:
        b_in, b_out = FAST_OPUS_BASE[model_name]
        base_in, out_p, cw_p, cr_p = b_in, b_out, b_in * 1.25, b_in * 0.1
    cw_1h_p = base_in * 2.0  # 1h cache write = 2x base, stacks on fast-mode base
    if (inference_geo or "").lower() == "us":
        base_in, out_p, cw_p, cr_p, cw_1h_p = (
            x * GEO_US_MULT for x in (base_in, out_p, cw_p, cr_p, cw_1h_p))
    cache_write = cw * cw_p
    if cw_1h > 0 and (cw_5m or 0) + cw_1h <= cw:
        cache_write += cw_1h * (cw_1h_p - cw_p)  # premium over the 5m rate
    return ((inp * base_in + out * out_p + cache_write + cr * cr_p) / 1_000_000
            + web_search * (WEB_SEARCH_PER_1K / 1000.0))
