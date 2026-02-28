"""Model pricing: static rates + LiteLLM dynamic fetch with caching."""

import json
import time
import urllib.request

from .config import LITELLM_URL, PRICING_CACHE_TTL
from .db import get_conn

MODEL_DISPLAY = {
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
    "Opus 4.6":  (5.00, 25.00, 6.25, 0.50),
    "Opus 4.5":  (5.00, 25.00, 6.25, 0.50),
    "Sonnet 4.6": (3.00, 15.00, 3.75, 0.30),
    "Sonnet 4.5": (3.00, 15.00, 3.75, 0.30),
    "Sonnet 4":   (3.00, 15.00, 3.75, 0.30),
    "Sonnet 3.5": (3.00, 15.00, 3.75, 0.30),
    "Haiku 4.5":  (1.00, 5.00, 1.25, 0.10),
    "Haiku 3.5":  (0.80, 4.00, 1.00, 0.08),
}
FALLBACK_PRICING = (3.00, 15.00, 3.75, 0.30)

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


def load_pricing():
    """Fetch Claude pricing from LiteLLM GitHub, with 24h DB-backed cache."""
    global _dynamic_pricing

    conn = get_conn()
    # Check DB cache
    row = conn.execute("SELECT value FROM meta WHERE key='pricing_cache'").fetchone()
    if row:
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


def get_pricing(model_name: str) -> tuple:
    if model_name in _dynamic_pricing:
        return _dynamic_pricing[model_name]
    return MODEL_PRICING.get(model_name, FALLBACK_PRICING)


def compute_cost(model_name: str, inp: int, out: int, cw: int, cr: int) -> float:
    p = get_pricing(model_name)
    return (inp * p[0] + out * p[1] + cw * p[2] + cr * p[3]) / 1_000_000
