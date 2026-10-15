"""GET /next - the new dashboard shell (Runway, Ledger, History, Lab).

The page is a static shell: every number comes from the JSON endpoints, which
the ES modules in static/next/js fetch. The server only decides the scope, the
owner name and the static version used to bust browser caches.
"""

import hashlib
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from . import config
from .auth import require_dashboard_auth
from .config import DEFAULT_SCOPE, STATS_OWNER, VALID_SCOPES

router = APIRouter()
_ROOT = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(_ROOT / "templates"))
_STATIC = _ROOT / "static" / "next"
DEFAULT_VIEW = "runway"


def _static_version(folder=_STATIC):
    """Short hash of the shell's static files (path, size, mtime).

    Computed once at import: a deploy restarts the process, so a new build
    gets a new version and the browser fetches main.js again.
    """
    digest = hashlib.sha1()
    if folder.is_dir():
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            stat = path.stat()
            digest.update(f"{path.relative_to(folder)}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:10]


STATIC_VERSION = _static_version()


def page_scope(requested, cookie_scope):
    """Same precedence as GET /: lock > ?scope= > tf_scope cookie > default.

    A bad or forbidden ?scope= soft-fails to the allowed scope (a bookmark must
    not 403 the page); the API routes still hard-fail with 400/403.
    """
    locked = config.LOCKED_SCOPE
    if locked and locked in VALID_SCOPES:
        return locked
    if requested in VALID_SCOPES:
        return requested
    if cookie_scope in VALID_SCOPES:
        return cookie_scope
    return DEFAULT_SCOPE


@router.get("/next", response_class=HTMLResponse, dependencies=[Depends(require_dashboard_auth)])
def next_dashboard(request: Request, scope: Optional[str] = None):
    effective = page_scope(scope, request.cookies.get("tf_scope"))
    response = templates.TemplateResponse(request, "next.html", {
        "scope": effective,
        "scope_locked": bool(config.LOCKED_SCOPE),
        "owner": STATS_OWNER,
        "default_view": DEFAULT_VIEW,
        "version": STATIC_VERSION,
    })
    # The security middleware sets no-store only on / and /api/*; this page
    # is behind the same auth, so a shared cache must not keep it either.
    response.headers["Cache-Control"] = "private, no-store"
    return response
