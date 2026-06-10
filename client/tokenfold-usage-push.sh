#!/bin/bash
# Stop/SessionEnd hook: immediately flush this machine's new Claude Code events to
# tokenfold, instead of waiting up to 5 min for the launchd/cron job. Detaches so the
# hook returns instantly and never blocks the session.
#
# TOKEN: TOKENFOLD_API_KEY must equal the server's STATS_API_KEY env var.
#   Dedicated token file (preferred): ~/.config/tokenfold-api-key
#   Fallback (back-compat):           ~/.config/notify-relay-token
# URL file: ~/.config/notify-relay-url  (shared with notify hooks)
#
# Auth failures are logged to ~/.tokenfold-push.log (stderr from the push script
# is redirected there so 401/403 messages are diagnosable).
HOOK_DIR="$(cd "$(dirname "$0")" && pwd)"
URL_FILE="$HOME/.config/notify-relay-url"
LOG_FILE="$HOME/.tokenfold-push.log"

# Prefer the dedicated ingest token; fall back to the notify-relay token for back-compat.
# The /api/ingest endpoint accepts only STATS_API_KEY (not the notify token), so machines
# whose notify token differs from STATS_API_KEY would 401 silently without this distinction.
if [ -f "$HOME/.config/tokenfold-api-key" ]; then
  TOKEN_FILE="$HOME/.config/tokenfold-api-key"
elif [ -f "$HOME/.config/notify-relay-token" ]; then
  TOKEN_FILE="$HOME/.config/notify-relay-token"
else
  TOKEN_FILE=""
fi

# Locate the push script: explicit override, next to this wrapper, or a launchd bin path.
PUSH=""
for cand in "$TOKENFOLD_PUSH" "$HOOK_DIR/tokenfold-push.py" \
            "/usr/local/bin/tokenfold-push.py" "$HOME/.local/bin/tokenfold-push.py"; do
  if [ -n "$cand" ] && [ -f "$cand" ]; then PUSH="$cand"; break; fi
done

# Nothing to do if config or push script is absent — exit cleanly (never fail the hook).
[ -f "$URL_FILE" ] && [ -n "$TOKEN_FILE" ] && [ -n "$PUSH" ] || exit 0

export TOKENFOLD_URL
TOKENFOLD_URL="$(cat "$URL_FILE")"
export TOKENFOLD_API_KEY
TOKENFOLD_API_KEY="$(cat "$TOKEN_FILE")"

# Debounce (for high-frequency hooks like PostToolUse): when
# TOKENFOLD_MIN_INTERVAL is set, skip if a push started in the last N seconds.
# Stamp races are benign — the server dedups by uuid.
if [ -n "$TOKENFOLD_MIN_INTERVAL" ]; then
  STAMP="$HOME/.tokenfold-last-push"
  now=$(date +%s)
  last=$(cat "$STAMP" 2>/dev/null || echo 0)
  [ $((now - last)) -lt "$TOKENFOLD_MIN_INTERVAL" ] && exit 0
  echo "$now" > "$STAMP"
fi

# Detach so the hook returns immediately. setsid on Linux, nohup fallback on macOS.
# Append to log file so auth failures (on stderr via the push script) are diagnosable.
if command -v setsid >/dev/null 2>&1; then
  setsid python3 "$PUSH" >>"$LOG_FILE" 2>&1 < /dev/null &
else
  nohup python3 "$PUSH" >>"$LOG_FILE" 2>&1 < /dev/null &
fi
exit 0
