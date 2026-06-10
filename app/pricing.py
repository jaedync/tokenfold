"""Model pricing: static rates + LiteLLM dynamic fetch with caching."""

import json
import time
import urllib.request

from .config import LITELLM_URL, PRICING_CACHE_TTL
from .db import get_conn

MODEL_DISPLAY = {
    "claude-fable-5": "Fable 5",
    "claude-opus-4-8": "Opus 4.8",
    "claude-opus-4-7": "Opus 4.7",
    "claude-opus-4-6": "Opus 4.6",
    "claude-opus-4-5-20251101": "Opus 4.5",
    "claude-sonnet-4-6": "Sonnet 4.6",
    "claude-sonnet-4-5-20250929": "Sonnet 4.5",
    "claude-sonnet-4-20250514": "Sonnet 4",
    "claude-haiku-4-5-20251001": "Haiku 4.5",
    "claude-3-5-sonnet-20241022": "Sonnet 3.5",
    "claude-3-5-haiku-20241022": "Haiku 3.5",
}

# Pricing per MTok: (input, output, cache_write_5m, cache_read)
MODEL_PRICING = {
    "Fable 5":   (10.00, 50.00, 12.50, 1.00),
    "Opus 4.8":  (5.00, 25.00, 6.25, 0.50),
    "Opus 4.7":  (5.00, 25.00, 6.25, 0.50),
    "Opus 4.6":  (5.00, 25.00, 6.25, 0.50),
    "Opus 4.5":  (5.00, 25.00, 6.25, 0.50),
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

# Canonical model sort order: Opus > Sonnet > Haiku, then version descending
MODEL_ORDER = list(MODEL_PRICING.keys())


def model_sort_key(name: str) -> int:
    try:
        return MODEL_ORDER.index(name)
    except ValueError:
        return len(MODEL_ORDER)  # unknown models sort last

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
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                    ("pricing_cache", json.dumps({"ts": time.time(), "pricing": pricing})),
                )
                conn.commit()
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


def get_pricing(model_name: str) -> tuple:
    if model_name in _dynamic_pricing:
        return _dynamic_pricing[model_name]
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]
    # Unknown: maybe LiteLLM knows it and our cache is just older than the
    # model — refresh once, then re-check. Otherwise it is unpriced ($0).
    _maybe_refresh_for_unknown(model_name)
    if model_name in _dynamic_pricing:
        return _dynamic_pricing[model_name]
    return _UNPRICED


def compute_cost(model_name: str, inp: int, out: int, cw: int, cr: int,
                 speed: str | None = None, inference_geo: str | None = None) -> float:
    """List-price cost. Optional speed='fast' (Opus only) and inference_geo='us'
    modifiers layer on the LiteLLM-or-static base rate; defaults reproduce prior
    pricing exactly. Mirrors claude-usage-telemetry's effective_rates()."""
    base_in, out_p, cw_p, cr_p = get_pricing(model_name)
    if (speed or "").lower() == "fast" and model_name in FAST_OPUS_BASE:
        b_in, b_out = FAST_OPUS_BASE[model_name]
        base_in, out_p, cw_p, cr_p = b_in, b_out, b_in * 1.25, b_in * 0.1
    if (inference_geo or "").lower() == "us":
        base_in, out_p, cw_p, cr_p = (x * GEO_US_MULT for x in (base_in, out_p, cw_p, cr_p))
    return (inp * base_in + out * out_p + cw * cw_p + cr * cr_p) / 1_000_000
