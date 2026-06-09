"""Two-tier auth: HTTP Basic for human dashboard routes, X-API-Key for machines."""
import hmac

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

import app.config as config

_basic = HTTPBasic(auto_error=False)


def require_dashboard_auth(credentials: HTTPBasicCredentials | None = Depends(_basic)):
    """Enforce HTTP Basic Auth on human-facing routes. Reads creds FRESH from
    app.config each call (so tests can monkeypatch). If DASHBOARD_PASSWORD is unset,
    auth is DISABLED (open) — a startup warning is logged in main.py."""
    password = config.DASHBOARD_PASSWORD
    if not password:
        return  # auth disabled (open) — see main.py startup warning
    user = config.DASHBOARD_USER or "admin"
    if credentials is None:
        _unauthorized()
    # constant-time compare BOTH fields; evaluate both to avoid early-out timing leaks
    u_ok = hmac.compare_digest(credentials.username, user)
    p_ok = hmac.compare_digest(credentials.password, password)
    if not (u_ok and p_ok):
        _unauthorized()


def _unauthorized():
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Basic"},  # makes the browser show its login dialog
    )


def require_api_key(x_api_key: str = Header(default="", alias="X-API-Key")):
    """Shared machine auth. Constant-time; fail-closed when STATS_API_KEY unset."""
    expected = config.STATS_API_KEY
    if not expected or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")
