import os

STATS_API_KEY = os.environ.get("STATS_API_KEY", "")
DB_PATH = os.environ.get("DB_PATH", "/app/data/tokenfold.db")
TZ_NAME = os.environ.get("TZ", "America/Chicago")
STATS_OWNER = os.environ.get("STATS_OWNER", "")
IDLE_THRESHOLD_S = 300
RECENCY_DAYS = 14
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
ORBB_IDLE_COLOR = None                  # 2200K warm white — "resting"
ORBB_IDLE_KELVIN = 2202                 # warmest the bulb supports
ORBB_TRANSITION = 3                     # seconds for gentle color fade
ORBB_SESSION_TTL = 300                  # seconds before stale session auto-expires
