"""GET /install.sh — the unauthenticated one-command install bootstrap.

Serves client/bootstrap.sh with the __TOKENFOLD_URL__ placeholder swapped for
the request's own external base URL, so

    curl -fsSL https://<server>/install.sh | bash -s -- --token 'tk_XXXX'

onboards a new machine against whatever host served the script. Unauthenticated
on purpose: the script carries ZERO secrets. The ingest token is passed by the
operator on the curl command line — it is never baked into the served body.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# Resolved repo-root-relative, the same way app/main.py resolves static/, so it
# works in the container and in local dev regardless of the process CWD. Module
# level (not per-request) so tests can monkeypatch it to exercise the 503 path.
BOOTSTRAP_PATH = Path(__file__).resolve().parent.parent / "client" / "bootstrap.sh"

# The placeholder inside bootstrap.sh, replaced with the live base URL at serve
# time. A raw copy fetched straight from GitHub keeps the placeholder and the
# script's own guard treats it as "no baked default".
_URL_PLACEHOLDER = "__TOKENFOLD_URL__"


def external_base_url(request: Request) -> str:
    """The public base URL the client used to reach us (e.g. 'https://usage.example.com').

    Single source of truth for both /install.sh (URL baked into the served
    script) and the dashboard's install-command button. The container runs
    uvicorn WITHOUT --proxy-headers, so request.url.scheme is always 'http'
    behind Caddy; trust Caddy's X-Forwarded-Proto for the real scheme, taking
    the FIRST value in case a proxy chain appended more hops. Direct local hits
    carry no such header and correctly fall back to the request scheme. No
    trailing slash — callers append their own path.
    """
    forwarded = request.headers.get("x-forwarded-proto")
    scheme = forwarded.split(",")[0].strip() if forwarded else request.url.scheme
    host = request.url.netloc  # Host header, including any :port
    return f"{scheme}://{host}"


@router.get("/install.sh", include_in_schema=False)
async def install_sh(request: Request) -> PlainTextResponse:
    try:
        script = BOOTSTRAP_PATH.read_text()
    except OSError:
        # A missing bootstrap.sh is a broken image, not a client error: log it
        # loudly (so monitoring notices) and 503 rather than 404 or a blank 200.
        logger.error("bootstrap script missing at %s", BOOTSTRAP_PATH, exc_info=True)
        raise HTTPException(status_code=503, detail="install script unavailable")

    baked = script.replace(_URL_PLACEHOLDER, external_base_url(request))
    return PlainTextResponse(
        baked,
        media_type="text/x-shellscript",
        headers={"Cache-Control": "no-store"},
    )
