#!/bin/bash
# Activity light hook — signals working/idle state to the relay service.
# Changes a Home Assistant light based on Claude Code session activity.
#
# Uses relay-post.sh for reliable delivery with retries.
# Requires: jq, md5sum (or md5 on macOS)

INPUT=$(cat)
HOOK_EVENT=$(echo "$INPUT" | jq -r '.hook_event_name // empty')

# For Stop hooks, skip if session is continuing (another hook is keeping it alive)
if [ "$HOOK_EVENT" = "Stop" ]; then
  STOP_HOOK_ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false')
  [ "$STOP_HOOK_ACTIVE" = "true" ] && exit 0
  STATE="idle"
else
  STATE="working"
fi

# Derive a short session ID from transcript_path
TRANSCRIPT=$(echo "$INPUT" | jq -r '.transcript_path // empty')
if [ -n "$TRANSCRIPT" ]; then
  if command -v md5sum >/dev/null 2>&1; then
    SESSION_ID=$(echo -n "$TRANSCRIPT" | md5sum | cut -c1-12)
  else
    SESSION_ID=$(echo -n "$TRANSCRIPT" | md5 | cut -c1-12)
  fi
else
  SESSION_ID="unknown"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"$SCRIPT_DIR/relay-post.sh" /api/light "{\"state\":\"$STATE\",\"session_id\":\"$SESSION_ID\"}"
