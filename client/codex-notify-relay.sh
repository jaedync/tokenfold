#!/bin/bash
# Codex CLI notification hook — thin client.
# Emits structured events to a notification relay service.
#
# Codex passes the JSON payload as a CLI argument (not stdin).
# Only fires on "agent-turn-complete" — no permission/approval events.
#
# Setup:
#   echo "https://your-tokenfold-server.example.com" > ~/.config/notify-relay-url
#   echo "<token>" > ~/.config/notify-relay-token
#
# Config (~/.codex/config.toml):
#   notify = ["bash", "<path-to>/codex-notify-relay.sh"]

RELAY_URL="${NOTIFY_RELAY_URL:-$(cat ~/.config/notify-relay-url 2>/dev/null)}"
RELAY_TOKEN="${NOTIFY_RELAY_TOKEN:-$(cat ~/.config/notify-relay-token 2>/dev/null)}"
[ -z "$RELAY_URL" ] || [ -z "$RELAY_TOKEN" ] && exit 0

INPUT="$1"
[ -z "$INPUT" ] && exit 0

EVENT_TYPE=$(echo "$INPUT" | jq -r '.type // empty')
CWD=$(echo "$INPUT" | jq -r '.cwd // empty')
PROJECT=$(basename "$CWD" 2>/dev/null || echo "unknown")

if [ "$EVENT_TYPE" = "agent-turn-complete" ]; then
  PAYLOAD=$(jq -n --arg p "$PROJECT" '{"event":"stop","project":$p}')
else
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"$SCRIPT_DIR/relay-post.sh" /api/notify "$PAYLOAD" || true
