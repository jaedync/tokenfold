"""GET /next - the new dashboard shell (Runway, Ledger, History, Lab).

The page is a static shell: every number comes from the JSON endpoints, which
the ES modules in static/next/js fetch. The server only decides the scope, the
owner name and the static version used to bust browser caches.
"""

import hashlib
import os
import re
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import config
from .auth import require_dashboard_auth
from .config import DEFAULT_SCOPE, STATS_OWNER, VALID_SCOPES

router = APIRouter()
_ROOT = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(_ROOT / "templates"))
_STATIC = _ROOT / "static" / "next"
DEFAULT_VIEW = "runway"
STATIC_PREFIX = "/static/next-"
_VERSION_RE = re.compile(r"^[0-9a-f]{6,40}$")
# The current version's files never change under their URL (a change makes
# a new version), so the browser may keep them for a year.
IMMUTABLE = "public, max-age=31536000, immutable"


def _files(folder):
    return sorted(p for p in folder.rglob("*") if p.is_file())


def _static_version(folder=_STATIC):
    """Short content hash of the shell's static files.

    Every module and asset URL carries it in its path
    (/static/next-<version>/js/...). Relative imports inherit the prefix, so
    a nested ES module cannot come from a stale browser cache after a
    deploy. A content hash (not mtimes) keeps the URLs, and the browser
    cache, across deploys that change nothing here.
    """
    digest = hashlib.sha1()
    if folder.is_dir():
        for path in _files(folder):
            digest.update(str(path.relative_to(folder)).encode() + b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()[:10]


def _signature(folder):
    return tuple((str(p), s.st_size, s.st_mtime_ns)
                 for p, s in ((p, p.stat()) for p in _files(folder)))


# Static imports and re-exports with a relative specifier. Dynamic import()
# is left out: it loads on demand.
_IMPORT_RE = re.compile(
    r"""(?:^|[;}\s])(?:import|export)\s*(?:[\w*{}\s,$]+?\s*from\s*)?['"](\.{1,2}/[^'"]+)['"]""")
_COMMENT_RE = re.compile(r"/\*.*?\*/|^\s*//.*?$", re.S | re.M)


def module_graph(folder, *entries):
    """Every module the entry modules import statically, as paths relative to
    folder, sorted. The shell lists them as modulepreload links, so the
    browser fetches the whole graph in one round instead of one round per
    import depth."""
    root = Path(folder).resolve()
    seen, todo = set(), list(entries)
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        path = (root / rel).resolve()
        if not path.is_file() or root not in path.parents:
            continue
        try:
            source = _COMMENT_RE.sub("", path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue  # not a module the browser can load; never fail the page
        seen.add(rel)
        for spec in _IMPORT_RE.findall(source):
            target = (path.parent / spec).resolve()
            # An import out of the folder (or a symlink out of it) is not a
            # static file this app serves: skip it, check before relative_to.
            if root in target.parents:
                todo.append(target.relative_to(root).as_posix())
    return sorted(seen)


_version_lock = threading.Lock()
_version_cache = {"sig": None, "version": None, "modules": ()}


def _refresh(folder):
    """Hash and scan again only when a file's size or mtime moved (a dev
    edit without a restart); else keep the cached version and module list."""
    try:
        sig = _signature(folder) if folder.is_dir() else ()
    except OSError:
        sig = None
    with _version_lock:
        if sig is None or sig != _version_cache["sig"]:
            _version_cache.update(sig=sig, version=_static_version(folder),
                                  modules=tuple(module_graph(folder, *ENTRY_MODULES)))
        return _version_cache["version"], _version_cache["modules"]


def static_version(folder=_STATIC):
    """The version for this page view."""
    return _refresh(folder)[0]


# The shell's module scripts (templates/next.html).
ENTRY_MODULES = ("js/early.js", "js/main.js")
STATIC_VERSION = static_version()
_static_files = StaticFiles(directory=str(_STATIC), check_dir=False)


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


# The shell and its modules are same-origin files and the panels style through
# CSSOM (el.style), never inline style or script, so the page needs no
# 'unsafe-inline'. img-src also allows data: URIs (none are used today).
CSP = "default-src 'self'; style-src 'self'; img-src 'self' data:"


# / is the dashboard; /next stays as an alias so bookmarks from before the
# swap keep working. The classic page lives at /classic (app/dashboard.py).
@router.get("/", response_class=HTMLResponse, dependencies=[Depends(require_dashboard_auth)])
@router.get("/next", response_class=HTMLResponse, dependencies=[Depends(require_dashboard_auth)])
def next_dashboard(request: Request, scope: Optional[str] = None):
    effective = page_scope(scope, request.cookies.get("tf_scope"))
    version, modules = _refresh(_STATIC)
    response = templates.TemplateResponse(request, "next.html", {
        "scope": effective,
        "scope_locked": bool(config.LOCKED_SCOPE),
        "owner": STATS_OWNER,
        "default_view": DEFAULT_VIEW,
        "version": version,
        "static_base": STATIC_PREFIX + version,
        "modules": modules,
    })
    # The security middleware sets no-store too; set here as well, so the
    # /next alias is covered whatever the middleware matches.
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["Content-Security-Policy"] = CSP
    return response


@router.api_route("/static/next-{version}/{path:path}", methods=["GET", "HEAD"],
                  include_in_schema=False)
async def versioned_static(version: str, path: str, request: Request):
    """static/next under a versioned path. StaticFiles resolves the file, so
    its traversal guard, ETag and range rules apply. A page from an older
    deploy still loads (the current files), but only the current version
    may be cached for good."""
    if not _VERSION_RE.fullmatch(version):
        raise HTTPException(status_code=404)
    rel = os.path.normpath(os.path.join(*path.split("/"))) if path else ""
    if not rel or rel.startswith(".."):
        raise HTTPException(status_code=404)
    response = await _static_files.get_response(rel, request.scope)
    if response.status_code in (200, 304):
        response.headers["Cache-Control"] = (
            IMMUTABLE if version == _version_cache["version"] else "no-cache")
    return response
