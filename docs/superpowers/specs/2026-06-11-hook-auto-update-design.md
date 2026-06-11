# Hook auto-update from GitHub — design

**Date:** 2026-06-11
**Decisions (Jaedyn):** full installer re-run · simple swap (no extra staging gates) · check on every debounced push.

## Problem

Client hook updates (pusher, wrapper, installer-level hook registration) currently
require manually re-running `install-tokenfold-hook.sh` on every machine. That is
burdensome and machines drift (e.g. enterprise machines still lack the PostToolUse
group and v2 backfill client).

## Mechanism

New `client/tokenfold-update.sh` (POSIX sh), installed to
`~/.claude/usage-telemetry/tokenfold-update.sh` by the installer.

1. **Trigger:** at the end of each debounced push, `tokenfold-usage-push.sh`
   spawns the updater fully detached (same detach pattern as the push). Zero
   added latency to the hook. `TOKENFOLD_NO_UPDATE=1` disables.
2. **Lock:** `mkdir`-based lock dir in `~/.claude/usage-telemetry/`; stale locks
   (>10 min old) are reclaimed. Concurrent pushes cannot double-update.
3. **Cheap change check:** GET
   `api.github.com/repos/jaedync/tokenfold/commits?path=client&per_page=1`
   (10s timeout, User-Agent set), parse the sha with python3, compare to the
   local stamp `~/.claude/usage-telemetry/.client-sha`. Same sha → exit.
   Any network/HTTP/parse failure → silent skip, retried on next push.
   Rate-limit math: 5-min push debounce ⇒ ≤12 checks/hr/machine vs GitHub's
   60/hr/IP unauthenticated budget.
4. **Update:** sha changed → download the tarball **pinned to that sha**
   (`codeload.github.com/jaedync/tokenfold/tar.gz/<sha>`), extract to a temp
   dir, and run the **downloaded** `client/install-tokenfold-hook.sh` from
   there (never the installed copy — avoids overwriting a running script).
   The installer is idempotent: copies scripts (incl. the updater itself),
   reconciles hook registration, backs up settings, resolves URL/token from
   `~/.config`, and verifies against prod with the sentinel POST.
5. **Commit:** installer exit 0 → write the new sha to the stamp. Non-zero →
   no stamp (retry next push); the previous scripts keep working.
6. **Logging:** append to `~/.claude/usage-telemetry/update.log`,
   self-truncated at ~100KB.

Endpoints are env-overridable (`TOKENFOLD_UPDATE_API`,
`TOKENFOLD_UPDATE_TARBALL_BASE`) so tests can use `file://` fixtures.

## Release flow

Pushing to `origin/main` = fleet rollout within one push cycle per active
machine. Before any push: grep all history for the scrubbed domain and launchd
label (repo policy). Note `origin/main` is currently stale; enabling the
feature includes pushing local `main`.

## Bootstrap & scope

- Each existing sh-hook machine needs ONE final manual installer run to receive
  the updater; after that, never again.
- Win11-Dev-VM (legacy Python pusher, no sh hook) is out of scope until it
  migrates to the hook.
- Server unchanged — this is client-only.

## Testing

- `app/tests/test_auto_update.py`, following `test_hook_scripts.py` patterns:
  - `sh -n` syntax check; source-level contracts (API path, sha-pinned tarball,
    lock, stamp, runs downloaded installer, no-update guard, wrapper spawns it,
    installer copies it).
  - Integration with `HOME` in a tempdir and `file://` fixtures: same-sha
    no-op; new-sha downloads + runs stub installer + writes stamp; installer
    failure → no stamp; held lock → skip.
- Full suite stays green.

## Failure posture

The updater can never break a push (spawned after, detached, all failures
silent-skip). Worst case a bad client lands on main: the installer's verify
step fails → stamp not written → machines retry, but scripts were already
copied by the broken run — recovery is pushing a fixed commit to main, which
machines pick up automatically (or restoring the installer's `.bak`s manually).
