#!/bin/bash
# Install Tokenfold Claude Code hooks on this machine.
#
# Usage: ./install-hooks.sh <server-url> <token>
#   e.g.: ./install-hooks.sh https://usage.example.com my-token-here
#
# What it does:
#   1. Saves relay URL and token to ~/.config/
#   2. Copies hook scripts to ~/.claude/hooks/
#   3. Merges hook entries into ~/.claude/settings.json
#
# Requires: jq

set -e

RELAY_URL="$1"
RELAY_TOKEN="$2"

if [ -z "$RELAY_URL" ] || [ -z "$RELAY_TOKEN" ]; then
  echo "Usage: $0 <server-url> <token>"
  echo "  e.g.: $0 https://usage.example.com my-token-here"
  exit 1
fi

command -v jq >/dev/null 2>&1 || { echo "Error: jq is required. Install it first."; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOOKS_DIR="$HOME/.claude/hooks"
SETTINGS="$HOME/.claude/settings.json"

# 1. Save relay config
mkdir -p "$HOME/.config"
echo "$RELAY_URL" > "$HOME/.config/notify-relay-url"
echo "$RELAY_TOKEN" > "$HOME/.config/notify-relay-token"
chmod 600 "$HOME/.config/notify-relay-token"
echo "[ok] Saved relay config to ~/.config/"

# 2. Copy hook scripts
mkdir -p "$HOOKS_DIR"
for script in relay-post.sh notify-relay.sh activity-light.sh codex-notify-relay.sh; do
  if [ -f "$SCRIPT_DIR/$script" ]; then
    cp "$SCRIPT_DIR/$script" "$HOOKS_DIR/$script"
    chmod +x "$HOOKS_DIR/$script"
  fi
done
echo "[ok] Copied hook scripts to $HOOKS_DIR/"

# 3. Merge hooks into settings.json
if [ ! -f "$SETTINGS" ]; then
  echo '{}' > "$SETTINGS"
fi

# Build the hooks we want to add
HOOKS_JSON=$(cat <<HOOKEOF
{
  "UserPromptSubmit": [
    {
      "hooks": [
        {
          "type": "command",
          "command": "$HOOKS_DIR/activity-light.sh",
          "timeout": 5
        }
      ]
    }
  ],
  "Stop": [
    {
      "hooks": [
        {
          "type": "command",
          "command": "$HOOKS_DIR/notify-relay.sh",
          "timeout": 10
        }
      ]
    },
    {
      "hooks": [
        {
          "type": "command",
          "command": "$HOOKS_DIR/activity-light.sh",
          "timeout": 5
        }
      ]
    }
  ],
  "Notification": [
    {
      "matcher": "permission_prompt",
      "hooks": [
        {
          "type": "command",
          "command": "$HOOKS_DIR/notify-relay.sh",
          "timeout": 10
        }
      ]
    },
    {
      "matcher": "elicitation_dialog",
      "hooks": [
        {
          "type": "command",
          "command": "$HOOKS_DIR/notify-relay.sh",
          "timeout": 10
        }
      ]
    }
  ]
}
HOOKEOF
)

# Merge: existing hooks take priority, our hooks fill in missing event types
EXISTING_HOOKS=$(jq '.hooks // {}' "$SETTINGS")
MERGED_HOOKS=$(echo "$EXISTING_HOOKS" | jq --argjson new "$HOOKS_JSON" '
  # For each event type in new, add it only if not already present
  reduce ($new | keys[]) as $key (
    .;
    if has($key) then . else .[$key] = $new[$key] end
  )
')

# Write back
jq --argjson hooks "$MERGED_HOOKS" '.hooks = $hooks' "$SETTINGS" > "$SETTINGS.tmp"
mv "$SETTINGS.tmp" "$SETTINGS"
echo "[ok] Updated $SETTINGS"

echo ""
echo "Done! Hooks installed. Restart Claude Code for changes to take effect."
echo ""
echo "To verify: claude --version && cat ~/.config/notify-relay-url"
