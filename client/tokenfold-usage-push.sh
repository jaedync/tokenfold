#!/bin/bash
# Stop/SessionEnd hook: immediately flush this machine's new Claude Code events to
# tokenfold, instead of waiting up to 5 min for the launchd/cron job. Detaches so the
# hook returns instantly and never blocks the session. Config is shared with the notify
# hooks (install-hooks.sh writes the notify-relay-url / notify-relay-token files).
HOOK_DIR="$(cd "$(dirname "$0")" && pwd)"
URL_FILE="$HOME/.config/notify-relay-url"
TOKEN_FILE="$HOME/.config/notify-relay-token"

# Locate the push script: explicit override, next to this wrapper, or a launchd bin path.
PUSH=""
for cand in "$TOKENFOLD_PUSH" "$HOOK_DIR/tokenfold-push.py" \
            "/usr/local/bin/tokenfold-push.py" "$HOME/.local/bin/tokenfold-push.py"; do
  if [ -n "$cand" ] && [ -f "$cand" ]; then PUSH="$cand"; break; fi
done

# Nothing to do if config or push script is absent — exit cleanly (never fail the hook).
[ -f "$URL_FILE" ] && [ -f "$TOKEN_FILE" ] && [ -n "$PUSH" ] || exit 0

export TOKENFOLD_URL="$(cat "$URL_FILE")"
export TOKENFOLD_API_KEY="$(cat "$TOKEN_FILE")"

# Detach so the hook returns immediately. setsid on Linux, nohup fallback on macOS.
if command -v setsid >/dev/null 2>&1; then
  setsid python3 "$PUSH" >/dev/null 2>&1 < /dev/null &
else
  nohup python3 "$PUSH" >/dev/null 2>&1 < /dev/null &
fi
exit 0
