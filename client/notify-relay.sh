#!/bin/bash
# Portable Claude Code notification hook — thin client.
# Emits structured events to a relay service that handles formatting/pricing.
#
# Setup on any machine:
#   echo "https://your-tokenfold-server.example.com" > ~/.config/notify-relay-url
#   echo "<token>" > ~/.config/notify-relay-token
#
# Or set env vars: NOTIFY_RELAY_URL, NOTIFY_RELAY_TOKEN

RELAY_URL="${NOTIFY_RELAY_URL:-$(cat ~/.config/notify-relay-url 2>/dev/null)}"
RELAY_TOKEN="${NOTIFY_RELAY_TOKEN:-$(cat ~/.config/notify-relay-token 2>/dev/null)}"
[ -z "$RELAY_URL" ] || [ -z "$RELAY_TOKEN" ] && exit 0

# Read hook input from stdin
INPUT=$(cat)
HOOK_EVENT=$(echo "$INPUT" | jq -r '.hook_event_name // empty')
CWD=$(echo "$INPUT" | jq -r '.cwd // empty')
PROJECT=$(basename "$CWD" 2>/dev/null || echo "unknown")
NOTIFICATION_TYPE=$(echo "$INPUT" | jq -r '.notification_type // empty')

if [ "$NOTIFICATION_TYPE" = "permission_prompt" ]; then
  PAYLOAD=$(jq -n --arg p "$PROJECT" '{"event":"permission","project":$p}')
elif [ "$NOTIFICATION_TYPE" = "elicitation_dialog" ]; then
  PAYLOAD=$(jq -n --arg p "$PROJECT" '{"event":"question","project":$p}')
elif [ "$HOOK_EVENT" = "Stop" ]; then
  # Skip if this is a re-fire from another stop hook continuing the session
  STOP_HOOK_ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false')
  [ "$STOP_HOOK_ACTIVE" = "true" ] && exit 0

  TRANSCRIPT=$(echo "$INPUT" | jq -r '.transcript_path // empty')

  # Wait for transcript to be flushed — the Stop hook fires before the
  # final assistant message is written to the transcript file.
  sleep 0.5

  if [ -n "$TRANSCRIPT" ] && [ -f "$TRANSCRIPT" ]; then
    # Find actual user prompt (string content), skip system/compaction messages
    LAST_USER_TS=$(tac "$TRANSCRIPT" | jq -r 'select(
      .type == "user" and
      (.message.content | type) == "string" and
      (.message.content | test("^\\s*<|^This session is being continued") | not)
    ) | .timestamp // empty' 2>/dev/null | head -1)

    # No genuine user prompt found = automated/bot session, skip notification
    [ -z "$LAST_USER_TS" ] && exit 0

    # Parse timestamp to epoch — macOS uses -j -f, Linux uses -d
    if date -d "$LAST_USER_TS" +%s >/dev/null 2>&1; then
      START_EPOCH=$(date -d "$LAST_USER_TS" +%s)
    elif date -j -f "%Y-%m-%dT%H:%M:%S" "${LAST_USER_TS%%.*}" +%s >/dev/null 2>&1; then
      START_EPOCH=$(date -j -f "%Y-%m-%dT%H:%M:%S" "${LAST_USER_TS%%.*}" +%s)
    else
      START_EPOCH=""
    fi

    END_EPOCH=$(date +%s)

    # Collect all transcript files (parent + subagents)
    TRANSCRIPT_FILES=("$TRANSCRIPT")
    SUBAGENT_DIR="${TRANSCRIPT%.jsonl}/subagents"
    if [ -d "$SUBAGENT_DIR" ]; then
      for f in "$SUBAGENT_DIR"/*.jsonl; do
        [ -f "$f" ] && TRANSCRIPT_FILES+=("$f")
      done
    fi

    # Extract models, tool count, and per-message usage from transcripts
    DURATION_JSON="${START_EPOCH:+$((END_EPOCH - START_EPOCH))}"
    DURATION_JSON="${DURATION_JSON:-null}"
    PAYLOAD=$(jq -r -n --arg ts "$LAST_USER_TS" --arg project "$PROJECT" \
      --argjson duration "$DURATION_JSON" \
    '
      reduce (inputs | select(.type == "assistant" and .timestamp > $ts and .message)) as $e (
        {tools: 0, models: [], usage: []};
        .tools += ([($e.message.content // [])[] | select(.type == "tool_use")] | length) |
        if $e.message.model then .models += [$e.message.model] else . end |
        if $e.message.usage then
          .usage += [{
            model: ($e.message.model // ""),
            input_tokens: ($e.message.usage.input_tokens // 0),
            output_tokens: ($e.message.usage.output_tokens // 0),
            cache_read_input_tokens: ($e.message.usage.cache_read_input_tokens // 0),
            cache_creation_input_tokens: ($e.message.usage.cache_creation_input_tokens // 0),
            cache_creation: ($e.message.usage.cache_creation // null)
          }]
        else . end
      ) | {
        event: "stop",
        project: $project,
        duration_s: $duration,
        models: (reduce .models[] as $m ([]; if (. | index($m)) then . else . + [$m] end)),
        tool_count: .tools,
        usage: .usage
      }
    ' "${TRANSCRIPT_FILES[@]}" 2>/dev/null)
  fi

  # If transcript parsing failed or wasn't available, send minimal stop event
  [ -z "$PAYLOAD" ] && PAYLOAD=$(jq -n --arg p "$PROJECT" '{"event":"stop","project":$p}')
else
  PAYLOAD=$(jq -n --arg p "$PROJECT" '{"event":"attention","project":$p}')
fi

curl -s -S --connect-timeout 3 --max-time 8 \
  -X POST "$RELAY_URL/api/notify" \
  -H "Authorization: Bearer $RELAY_TOKEN" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD" >/dev/null 2>&1 || true
