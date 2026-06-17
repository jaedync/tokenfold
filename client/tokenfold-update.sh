#!/bin/sh
# tokenfold-update.sh — self-update the installed hook client from GitHub.
#
# Spawned DETACHED by tokenfold-usage-push.sh after each debounced push, so it
# can never slow down or fail a hook. Cheap check first: the latest commit sha
# touching client/ on main (one small GitHub API call) vs the local stamp.
# On change, download the tarball PINNED to that sha and re-run the DOWNLOADED
# install-tokenfold-hook.sh — never the installed copy, which would overwrite
# a script while it executes. The installer is idempotent: it copies the hook
# scripts (including this one), reconciles hook registration in settings.json,
# resolves URL/token from ~/.config, and verifies against the server. The
# stamp only advances when the installer exits 0, so failures retry on the
# next push while the existing scripts keep working.
#
# Kill switch: TOKENFOLD_NO_UPDATE=1
# Test seams:  TOKENFOLD_UPDATE_API / TOKENFOLD_UPDATE_TARBALL_BASE (file:// ok)

[ -n "$TOKENFOLD_NO_UPDATE" ] && exit 0

HOOKS_DIR="$HOME/.claude/usage-telemetry"
STAMP="$HOOKS_DIR/.client-sha"
LOCK="$HOOKS_DIR/.update-lock"
LOG="$HOOKS_DIR/update.log"
API="${TOKENFOLD_UPDATE_API:-https://api.github.com/repos/jaedync/tokenfold/commits?path=client&per_page=1}"
TARBALL_BASE="${TOKENFOLD_UPDATE_TARBALL_BASE:-https://codeload.github.com/jaedync/tokenfold/tar.gz}"

mkdir -p "$HOOKS_DIR" 2>/dev/null

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

# Keep the log bounded (~100KB cap, halved when hit).
if [ -f "$LOG" ] && [ "$(wc -c < "$LOG" 2>/dev/null || echo 0)" -gt 100000 ]; then
  tail -c 50000 "$LOG" > "$LOG.tmp" 2>/dev/null && mv "$LOG.tmp" "$LOG"
fi

# Single-flight lock. A fresh lock means an update is in progress — bail.
# A stale lock (>10 min) is a crashed prior run — reclaim it.
if ! mkdir "$LOCK" 2>/dev/null; then
  if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +10 2>/dev/null)" ]; then
    rm -rf "$LOCK"
    mkdir "$LOCK" 2>/dev/null || exit 0
  else
    exit 0
  fi
fi
TMPD=""
trap 'rm -rf "$LOCK" "$TMPD"' EXIT

# Latest client/ commit sha. Any network/HTTP/parse failure = silent skip.
sha=$(curl -fsS -m 10 -H "User-Agent: tokenfold-update" "$API" 2>/dev/null \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["sha"])' 2>/dev/null)
# Strict lowercase-hex guard: the sha is interpolated into a URL, so anything
# else (API error body, tampered response) must never leave this line.
case "$sha" in
  "") exit 0 ;;
  *[!0-9a-f]*) exit 0 ;;
esac
[ "$sha" = "$(cat "$STAMP" 2>/dev/null)" ] && exit 0

log "new client sha $sha (local: $(cat "$STAMP" 2>/dev/null || echo none))"

TMPD=$(mktemp -d) || exit 0
if ! curl -fsSL -m 120 -H "User-Agent: tokenfold-update" "$TARBALL_BASE/$sha" 2>/dev/null \
    | tar -xzf - -C "$TMPD" 2>/dev/null; then
  log "download/extract failed — will retry next push"
  exit 0
fi

installer=$(find "$TMPD" -name install-tokenfold-hook.sh -path '*/client/*' 2>/dev/null | head -1)
if [ -z "$installer" ]; then
  log "installer missing from tarball — will retry next push"
  exit 0
fi

# TOKENFOLD_NO_UPDATE guards against any recursive update spawn from inside
# the installer's verification push.
if TOKENFOLD_NO_UPDATE=1 bash "$installer" >> "$LOG" 2>&1; then
  printf '%s' "$sha" > "$STAMP"
  log "client updated to $sha"
else
  log "installer failed — stamp not advanced, will retry next push"
fi
exit 0
