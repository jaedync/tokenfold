# One-Command Install: `/install.sh` + Dashboard Copy Button — Design

**Date:** 2026-07-01 · **Status:** Approved by Jaedyn (chat) · **Target:** local `main` @ `8dd16fe`+

## Problem

Installing the usage hook on a new machine today requires: clone the repo, `cd client/`,
run `./install-tokenfold-hook.sh --url <SERVER_URL> --token <STATS_API_KEY>`. The dashboard
footer has a copy-ingest-key button, but the rest of the process is manual. We want
claude.ai-style onboarding: one copyable command that does everything.

## Approved shape

```bash
curl -fsSL https://usage.jaedynchilton.com/install.sh | bash -s -- --token 'tk_XXXX'
```

- The app serves `/install.sh` (unauthenticated — the script contains **zero secrets**).
- The served script has the server's own base URL baked in at request time.
- A new footer button next to **Ingest Key** copies the full command (with the real key
  inline) to the clipboard. Fail-closed: rendered only when `DASHBOARD_PASSWORD` is set,
  exactly like `ingest_key`.

## Components

### 1. `client/bootstrap.sh` (new)

Bash (piped into `bash`, matching the recent "run installer via bash, not sh" fix).
Responsibilities, in order:

1. **Arg handling:** recognize `--url <v>` and `--token <v>`; pass every other arg through
   to the installer verbatim (`--no-push`, `--keep-legacy`, `--verify-only`, ...).
   Token may be omitted (installer resolves `$TOKENFOLD_API_KEY` / `~/.config` itself).
2. **Baked URL:** constant `TOKENFOLD_URL_DEFAULT="__TOKENFOLD_URL__"` substituted by the
   server. If it still starts with `__` (script fetched raw from GitHub, unsubstituted),
   treat as unset. Effective URL = `--url` arg > baked default. If neither → die with a
   clear message. Pass as `--url` to the installer (prepend, so a user-supplied `--url`
   later in argv wins — the installer's parser lets the last occurrence overwrite).
3. **Preflight:** `curl`, `tar`, `python3` on PATH, else die with which one is missing.
4. **Fetch:** download `https://codeload.github.com/jaedync/tokenfold/tar.gz/refs/heads/main`
   to `mktemp -d` (cleanup `trap` on EXIT). No GitHub API call, no sha pinning — the
   installed self-updater pins shas from then on. Override seam for tests:
   `TOKENFOLD_BOOTSTRAP_TARBALL` (full URL, `file://` OK), mirroring the self-updater's
   `TOKENFOLD_UPDATE_TARBALL_BASE`.
5. **Extract + run:** `tar -xzf`, locate `*/client/install-tokenfold-hook.sh` by glob,
   `bash <that> --url <effective> [passthrough args...]`. Exit with the installer's code.
6. **Tone:** brief `[bootstrap]`-prefixed progress lines; the installer owns the real
   output. Errors go to stderr, non-zero exit.

### 2. `app/install.py` (new, small router)

- `external_base_url(request) -> str` — scheme = first value of `X-Forwarded-Proto` if
  present else `request.url.scheme`; host = `Host` header (`request.url.netloc`). Returns
  `f"{scheme}://{host}"`, no trailing slash. (Container runs uvicorn **without**
  `--proxy-headers`; Caddy sets `X-Forwarded-Proto: https`. Local hits fall back to http.)
  This helper is imported by `app/dashboard.py` too — single source of truth.
- `GET /install.sh` (no auth dependency, `include_in_schema=False`): read
  `client/bootstrap.sh` from the repo root (path resolved like `app/main.py` resolves
  `static/`), replace `__TOKENFOLD_URL__` → `external_base_url(request)`, return
  `PlainTextResponse` with `media_type="text/x-shellscript"` and
  `Cache-Control: no-store`. If the file is missing (broken image) → HTTP 503 with a
  short detail, and log the error server-side.
- Wire into `app/main.py` via `app.include_router(...)` next to the existing routers.

### 3. Dashboard command + button

- `app/dashboard.py`: new helper `build_install_command(base_url, key) -> str` producing
  `curl -fsSL {base}/install.sh | bash -s -- --token {shlex.quote(key)}`. Template context
  gains `install_cmd`: built only when `config.DASHBOARD_PASSWORD` and
  `config.STATS_API_KEY` are both set, else `""` (mirrors `ingest_key` fail-closed
  comment — keep a matching why-comment).
- `templates/dashboard.html` footer: inside the existing `{% if ingest_key %}` block, add
  `{% if install_cmd %}` button **Install Command** immediately after the Ingest Key
  button (separated by the existing `·` pattern), reusing `.ingest-key-btn` styling.
  `data-install-cmd="{{ install_cmd|e }}"`, `title` explains: copies a curl-pipe-bash
  one-liner (contains the ingest key) for setting up a new machine.
  Click = clipboard copy + transient ` copied` note (same pattern as the key button),
  **no inline reveal** (command is ~90 chars). Keyboard-operable by construction
  (`<button type="button">`). If `navigator.clipboard` is unavailable, fall back to
  revealing the command text in a `<span>` so the user can copy manually (the existing
  key button's reveal-first behavior makes clipboard optional; a copy-only button must
  not silently no-op).

## Tests (repo convention: `unittest` in `app/tests/`, FastAPI `TestClient`; RED → GREEN)

- `app/tests/test_install_sh.py`:
  - 200, body contains `codeload.github.com/jaedync/tokenfold`, no `__TOKENFOLD_URL__`
    remnant, baked URL matches request host; honors `X-Forwarded-Proto: https`.
  - Response body never contains `STATS_API_KEY` value.
  - No auth required (no 401 without credentials).
  - 503 when bootstrap file missing (monkeypatch path).
- Extend the existing footer/template tests (see `app/tests/test_footer_token.py` and
  `test_dashboard_template.py` for the established pattern):
  - `install_cmd` present + button rendered when `DASHBOARD_PASSWORD` set; absent when not.
  - `build_install_command` quotes a key containing a single quote safely.
- `client/test_bootstrap.py` (stdlib `unittest`, pattern of `client/test_desktop_metadata.py`):
  build a `file://` tarball containing a **stub** `client/install-tokenfold-hook.sh` that
  echoes its argv; assert URL baking, `--url` override order, arg passthrough, tmpdir
  cleanup, non-zero exit + message when no URL resolvable.

Run: `.venv/bin/python -m unittest discover -s app/tests -v` and
`.venv/bin/python -m unittest client.test_bootstrap` — **all pre-existing tests must
still pass.**

## Accepted risks / non-goals

- The copied command embeds the live ingest key → shell history on the target machine.
  Same exposure class as the existing copy-key button; accepted.
- `/install.sh` serves the image-baked bootstrap (may lag GitHub `main`); irrelevant in
  practice because the bootstrap immediately fetches latest `main` anyway.
- Deploy to ms01 is out of scope here; it rides the audit-template reconciliation deploy
  (do NOT `docker cp` this one — it spans app code, template, and client files).
