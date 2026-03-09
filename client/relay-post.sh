#!/bin/bash
# Shared HTTP POST helper for Claude Code hooks.
# Retries up to 3 times with 1s backoff. Logs failures to stderr.
#
# Usage: relay-post.sh <endpoint-path> <json-payload>
#   e.g.: relay-post.sh /api/light '{"state":"working"}'
#
# Config:
#   Env vars: NOTIFY_RELAY_URL, NOTIFY_RELAY_TOKEN
#   Or files: ~/.config/notify-relay-url, ~/.config/notify-relay-token

RELAY_URL="${NOTIFY_RELAY_URL:-$(cat ~/.config/notify-relay-url 2>/dev/null)}"
RELAY_TOKEN="${NOTIFY_RELAY_TOKEN:-$(cat ~/.config/notify-relay-token 2>/dev/null)}"
[ -z "$RELAY_URL" ] || [ -z "$RELAY_TOKEN" ] && exit 0

ENDPOINT="$1"
PAYLOAD="$2"

for attempt in 1 2 3; do
  if curl -s -f --connect-timeout 3 --max-time 8 \
    -X POST "$RELAY_URL$ENDPOINT" \
    -H "Authorization: Bearer $RELAY_TOKEN" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" >/dev/null 2>&1; then
    exit 0
  fi
  [ "$attempt" -lt 3 ] && sleep 1
done

echo "[relay-post] Failed after 3 attempts: $ENDPOINT" >&2
exit 1
