import os

STATS_API_KEY = os.environ.get("STATS_API_KEY", "")
DB_PATH = os.environ.get("DB_PATH", "/app/data/tokenfold.db")
TZ_NAME = os.environ.get("TZ", "America/Chicago")
STATS_OWNER = os.environ.get("STATS_OWNER", "")
IDLE_THRESHOLD_S = 300
RECENCY_DAYS = 14
LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
PRICING_CACHE_TTL = 86400  # 24 hours
