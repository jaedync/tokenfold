import os

STATS_API_KEY = os.environ.get("STATS_API_KEY", "")
DB_PATH = os.environ.get("DB_PATH", "/app/data/tokenfold.db")
TZ_NAME = os.environ.get("TZ", "America/Chicago")
STATS_OWNER = os.environ.get("STATS_OWNER", "")
IDLE_THRESHOLD_S = 300
RECENCY_DAYS = 14

# Verified-enterprise membership as a TOTAL boolean (never NULL under SQL 3-valued
# logic), so PERSONAL_PRED = NOT(ENTERPRISE_PRED) is a true partition: every row is
# in exactly one scope and the two scopes' totals sum to the blended total. NULL/empty
# plan, org, or account => NOT enterprise => personal. (Plan is the signal; non-empty
# org + a real account email are defense-in-depth — an unattributed 'enterprise' row
# is personal.) Logically identical TRUE-set to the prior predicate (NULL and FALSE
# both fail a WHERE), so existing enterprise results are unchanged.
ENTERPRISE_PRED = (
    "COALESCE(plan,'') = 'enterprise' "
    "AND COALESCE(org_name,'') != '' "
    "AND COALESCE(account_email,'') NOT IN ('', 'unknown')"
)
PERSONAL_PRED = "NOT (" + ENTERPRISE_PRED + ")"

VALID_SCOPES = ("enterprise", "personal")
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
