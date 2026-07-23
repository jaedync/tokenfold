import os

STATS_API_KEY = os.environ.get("STATS_API_KEY", "")
DB_PATH = os.environ.get("DB_PATH", "/app/data/tokenfold.db")
DASHBOARD_USER = os.environ.get("DASHBOARD_USER", "admin")
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")  # unset => dashboard auth disabled (open)
TZ_NAME = os.environ.get("TZ", "America/Chicago")
STATS_OWNER = os.environ.get("STATS_OWNER", "")
IDLE_THRESHOLD_S = 300
RECENCY_DAYS = 14

# Agent-state / notification policy (see agent_state.py, notify.py).
# A session that stops reporting decays out of the store after this TTL,
# so a killed terminal can never strand a waiting state.
AGENT_STATE_TTL_S = int(os.environ.get("AGENT_STATE_TTL_S", "600"))
# Waiting sessions keep a much longer leash: a blocked permission prompt is
# still genuinely waiting after 10 minutes, and pruning it made the cube
# read idle while Claude sat on a tool-use approval. Cleanly-closed
# terminals delete themselves via the SessionEnd "gone" event, so this
# long TTL only decides how long a hard-crashed terminal can strand a red
# beacon.
AGENT_STATE_WAITING_TTL_S = int(os.environ.get("AGENT_STATE_WAITING_TTL_S", "7200"))
# Fan-out mote leash: a subagent with no SubagentStop decays after this TTL,
# so a missed stop cannot strand a mote on the cube. Active subagents are
# kept fresh by their parent's working heartbeats (background subagents let
# the parent keep emitting PostToolUse), so this only bounds stranded ones.
AGENT_STATE_SUBAGENT_TTL_S = int(os.environ.get("AGENT_STATE_SUBAGENT_TTL_S", "180"))
# A session counts as "actively grinding" (vs "handed off to subagents") only
# while its last GENUINE working heartbeat is fresher than this. Longer than
# the relay's 60s PostToolUse heartbeat throttle so a steadily-working session
# never flickers out of the working mood between heartbeats.
AGENT_STATE_WORKING_FRESH_S = int(os.environ.get("AGENT_STATE_WORKING_FRESH_S", "90"))
# Minimum time a fan-out mote stays on the roster after spawn, even if its
# SubagentStop arrives almost immediately, so an ambient display polling every
# ~2s actually sees (and can flash) a very short-lived subagent.
AGENT_STATE_MOTE_MIN_VISIBLE_S = float(os.environ.get("AGENT_STATE_MOTE_MIN_VISIBLE_S", "3.0"))
# How long a stopped mote lingers as "sunsetting" so the cube can play its
# death flash before it is dropped.
AGENT_STATE_MOTE_SUNSET_S = float(os.environ.get("AGENT_STATE_MOTE_SUNSET_S", "0.8"))
# Delayed stop receipts: an interactive stop never pushes "Response complete"
# immediately. The Stop hook fires at the end of every main-loop turn (including
# turns that only hand off to a subagent or background task), so the receipt is
# held for this quiet window and pushed only if the session stayed idle. A
# receipt that arrives ~25s late is fine for a phone cost receipt.
RECEIPT_QUIET_S = float(os.environ.get("RECEIPT_QUIET_S", "25.0"))
# Presence damping: suppress waiting pushes when any session got a user
# prompt this recently (someone is at a keyboard; the ambient display
# still shows the beacon). 0 disables damping.
AGENT_PRESENCE_DAMPING_S = int(os.environ.get("AGENT_PRESENCE_DAMPING_S", "120"))
# Trouble overlay leash: a tool failure (PostToolUseFailure / StopFailure)
# marks the session troubled for this long, then the overlay self-clears even
# without a clearing event. Fresh tool progress or a new prompt clears it early.
TROUBLE_TTL_S = float(os.environ.get("AGENT_STATE_TROUBLE_TTL_S", "45"))
# Compaction overlay fallback: if PostCompact (compact_end) is dropped, the
# compacting overlay self-clears this long after compact_start so a missed end
# cannot strand a session showing "compacting" forever.
COMPACT_TTL_S = float(os.environ.get("AGENT_STATE_COMPACT_TTL_S", "300"))


def _csv_env(name, default=""):
    return tuple(x.strip() for x in os.environ.get(name, default).split(",") if x.strip())


# Enterprise signals (all optional, OR'd). Fail-closed: anything not positively
# matched is PERSONAL. Configurable per-instance without code changes.
#
# ENTERPRISE_PRED is a TOTAL boolean (never NULL under SQL 3-valued logic), so
# PERSONAL_PRED = NOT(ENTERPRISE_PRED) is a true partition: every row is in exactly
# one scope and the two scopes' totals sum to the blended total. NULL/empty plan,
# org_type, org_uuid, or account => NOT enterprise => personal.
#
# The old org_name != '' guard is intentionally DROPPED: org_name is populated
# even on personal accounts (it is not a reliable enterprise signal). The
# account_email guard stays — an unattributed 'enterprise' row is still personal.
ENTERPRISE_ORG_TYPES = _csv_env("ENTERPRISE_ORG_TYPES", "claude_enterprise,claude_team")
ENTERPRISE_ORG_UUIDS = _csv_env("ENTERPRISE_ORG_UUIDS")
ENTERPRISE_EMAIL_DOMAINS = _csv_env("ENTERPRISE_EMAIL_DOMAINS")

# US-residency assumption: Claude Code transcripts stamp inference_geo
# 'not_available' on ALL subscription traffic, so a US-pinned workspace
# (1.1x on every token category, Opus/Sonnet 4.6+) is invisible to us.
# TOKENFOLD_ENTERPRISE_GEO=us bills enterprise-classified usage at the US
# rate at COMPUTE TIME — raw events are never modified; revert by clearing
# the env (stored daily rollups need a re-roll either way).
_assume_geo = os.environ.get("TOKENFOLD_ENTERPRISE_GEO", "").strip().lower()
if _assume_geo not in ("", "us"):
    print(f"[config] WARNING: TOKENFOLD_ENTERPRISE_GEO={_assume_geo!r} "
          "unsupported (only 'us') — ignored")
    _assume_geo = ""
ENTERPRISE_ASSUME_GEO = _assume_geo


def _sql_in(col, values):
    quoted = ",".join("'" + v.replace("'", "''") + "'" for v in values)
    return f"COALESCE({col},'') IN ({quoted})"


_signals = ["COALESCE(plan,'') = 'enterprise'"]
if ENTERPRISE_ORG_TYPES:
    _signals.append(_sql_in("org_type", ENTERPRISE_ORG_TYPES))
if ENTERPRISE_ORG_UUIDS:
    _signals.append(_sql_in("org_uuid", ENTERPRISE_ORG_UUIDS))
for _dom in ENTERPRISE_EMAIL_DOMAINS:
    _d = _dom.lstrip("@")
    # LIKE wildcard hardening: '%' and '_' in a configured domain are live SQL
    # LIKE wildcards (e.g. '%.io' would classify EVERY *.io personal account as
    # enterprise). Skip such domains entirely (fail-closed) rather than strip:
    # stripping 'my_corp.io' -> 'mycorp.io' would silently match a DIFFERENT
    # domain, which is worse than not matching at all.
    if "%" in _d or "_" in _d:
        print(f"[config] WARNING: ENTERPRISE_EMAIL_DOMAINS entry {_dom!r} contains "
              f"SQL LIKE wildcard characters (%/_) — ignored (fail-closed)")
        continue
    _d = _d.replace("'", "''")
    _signals.append(f"COALESCE(account_email,'') LIKE '%@{_d}'")

ENTERPRISE_PRED = (
    "(COALESCE(account_email,'') NOT IN ('', 'unknown') AND ("
    + " OR ".join(_signals) + "))"
)
PERSONAL_PRED = "NOT " + ENTERPRISE_PRED

VALID_SCOPES = ("enterprise", "personal")
# Which scope the dashboard lands on / APIs fall back to. Configurable per-instance:
# a personal-only deployment (e.g. ms01) sets TOKENFOLD_DEFAULT_SCOPE=personal so the
# dashboard doesn't open on an empty enterprise view. Invalid values fall back safely.
DEFAULT_SCOPE = os.environ.get("TOKENFOLD_DEFAULT_SCOPE", "enterprise")
if DEFAULT_SCOPE not in VALID_SCOPES:
    DEFAULT_SCOPE = "enterprise"

# Env lock: when set to a valid scope, the instance is LOCKED to it — the UI toggle is
# hidden and the API refuses other scopes (fail-closed compliance posture).
LOCKED_SCOPE = os.environ.get("TOKENFOLD_SCOPE") or None


class InvalidScope(ValueError):
    pass


class ScopeLocked(Exception):
    pass


def scope_predicate(scope: str) -> str:
    """Return the SQL predicate string for the given scope name."""
    if scope == "enterprise":
        return ENTERPRISE_PRED
    if scope == "personal":
        return PERSONAL_PRED
    raise InvalidScope(f"invalid scope: {scope!r}")


def resolve_scope(requested):
    """Effective scope for an API request. Reads the module-level LOCKED_SCOPE FRESH
    each call (so tests can monkeypatch app.config.LOCKED_SCOPE). Raises InvalidScope
    (-> 400) for an unknown scope, ScopeLocked (-> 403) when a locked instance is asked
    for a different scope. With no lock, falls back to DEFAULT_SCOPE."""
    import app.config as cfg
    locked = cfg.LOCKED_SCOPE
    if requested is not None and requested not in VALID_SCOPES:
        raise InvalidScope(f"invalid scope: {requested!r}")
    if locked:
        if locked not in VALID_SCOPES:
            raise InvalidScope(f"invalid TOKENFOLD_SCOPE lock: {locked!r}")
        if requested is not None and requested != locked:
            raise ScopeLocked(f"instance is locked to scope {locked!r}")
        return locked
    return requested or DEFAULT_SCOPE
LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
PRICING_CACHE_TTL = 86400  # 24 hours

# OAuth credentials file (mounted from host)
CLAUDE_CREDENTIALS_PATH = os.environ.get("CLAUDE_CREDENTIALS_PATH", "/app/credentials.json")

# Notify relay (optional — all empty = feature disabled)
NOTIFY_TOKEN = os.environ.get("NOTIFY_TOKEN", "")
HA_URL = os.environ.get("HA_URL", "")
HA_TOKEN = os.environ.get("HA_TOKEN", "")
HA_DEVICES = [d.strip() for d in os.environ.get("HA_DEVICES", "").split(",") if d.strip()]

# ORBB activity light (optional — all empty = feature disabled)
ORBB_ENTITY = os.environ.get("ORBB_ENTITY", "light.orbb")
ORBB_WORKING_COLOR = [255, 122, 10]    # Claude orange — "thinking"
ORBB_WORKING_BRIGHTNESS = 204          # 80%
ORBB_IDLE_COLOR = None                  # 2200K warm white — "resting"
ORBB_IDLE_KELVIN = 2202                 # warmest the bulb supports
ORBB_IDLE_BRIGHTNESS = 51              # 20%
ORBB_TRANSITION = 3                     # seconds for gentle color fade
ORBB_SESSION_TTL = 300                  # seconds before stale session auto-expires
