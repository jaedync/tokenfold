#!/usr/bin/env bash
# install-tokenfold-hook.sh — idempotent installer + verifier for the tokenfold
# Stop/SessionEnd usage hook ("phone home" to your tokenfold server).
#
# Point a Claude Code session at client/HOOK-SETUP.md and it will run this.
# Safe to re-run: a second run is a no-op except for re-verification.
#
# Usage:
#   ./install-tokenfold-hook.sh --url <SERVER_URL> --token <STATS_API_KEY> [options]
#   ./install-tokenfold-hook.sh --verify-only        # check an existing install
#
# Nothing server-identifying is hardcoded: both the URL and the token are inputs.
# URL resolution (first hit wins):
#   1. --url <value>
#   2. $TOKENFOLD_URL in the environment
#   3. ~/.config/notify-relay-url    (existing install)
# Token resolution (first hit wins):
#   1. --token <value>
#   2. $TOKENFOLD_API_KEY in the environment
#   3. ~/.config/tokenfold-api-key   (dedicated ingest token, preferred)
#   4. ~/.config/notify-relay-token  (legacy shared notify token)
#
# Options:
#   --url <URL>      Server base URL (required unless already configured)
#   --token <TOKEN>  STATS_API_KEY for /api/ingest auth
#   --verify-only    Skip install; only run the auth probe against current config
#   --no-push        Install + verify auth, but do NOT fire a real push
#   --cleanup-only   Only retire the legacy launchd/cron reporter, then exit
#   --keep-legacy    Do NOT touch any pre-existing launchd/cron reporter
#   --watch          Also install the resident watcher LaunchAgent (macOS-only):
#                    events land within ~1s instead of on the next hook fire
#   -h | --help      Show this help
#
# This hook REPLACES the old periodic reporter (a claude-stats-push launchd
# agent / a cron job running claude-stats-push.py).
# Running both would mean two reporters on one machine, and the old one predates
# org_type/org_uuid attribution (it would land enterprise usage as personal). By
# default the installer retires the legacy reporter (backing everything up first).
# Use --keep-legacy to opt out.
#
# Exit codes: 0 = success (installed and/or verified), non-zero = failure.
set -euo pipefail

HOOKS_DIR="$HOME/.claude/usage-telemetry"
CONFIG_DIR="$HOME/.config"
SETTINGS="$HOME/.claude/settings.json"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

URL=""
TOKEN=""
VERIFY_ONLY=0
DO_PUSH=1
CLEANUP_ONLY=0
KEEP_LEGACY=0
INSTALL_WATCH=0

WATCH_PLIST_SRC="$SCRIPT_DIR/com.jaedynchilton.tokenfold-watch.plist"
WATCH_LABEL="com.jaedynchilton.tokenfold-watch"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"

log()  { printf '  %s\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --url)         URL="${2:-}"; shift 2 ;;
    --token)       TOKEN="${2:-}"; shift 2 ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    --no-push)     DO_PUSH=0; shift ;;
    --cleanup-only) CLEANUP_ONLY=1; shift ;;
    --keep-legacy) KEEP_LEGACY=1; shift ;;
    --watch)       INSTALL_WATCH=1; shift ;;
    -h|--help)     sed -n '2,43p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)             die "unknown argument: $1 (use --help)" ;;
  esac
done

command -v python3 >/dev/null 2>&1 || die "python3 is required but not found on PATH"
command -v curl    >/dev/null 2>&1 || die "curl is required but not found on PATH"

# ── Resolve URL ───────────────────────────────────────────────────────────────
if [ -z "$URL" ]; then URL="${TOKENFOLD_URL:-}"; fi
if [ -z "$URL" ] && [ -f "$CONFIG_DIR/notify-relay-url" ]; then
  URL="$(cat "$CONFIG_DIR/notify-relay-url")"
fi
[ -n "$URL" ] || die "no server URL. Pass --url <SERVER_URL>, set \$TOKENFOLD_URL, or have $CONFIG_DIR/notify-relay-url from a prior install"
URL="${URL%/}"  # strip trailing slash

# ── Resolve token (never logged) ──────────────────────────────────────────────
if [ -z "$TOKEN" ]; then TOKEN="${TOKENFOLD_API_KEY:-}"; fi
if [ -z "$TOKEN" ] && [ -f "$CONFIG_DIR/tokenfold-api-key" ]; then
  TOKEN="$(cat "$CONFIG_DIR/tokenfold-api-key")"
fi
if [ -z "$TOKEN" ] && [ -f "$CONFIG_DIR/notify-relay-token" ]; then
  TOKEN="$(cat "$CONFIG_DIR/notify-relay-token")"
fi

echo "tokenfold hook installer"
echo "  server: $URL"

# ── Legacy reporter cleanup ───────────────────────────────────────────────────
# Retire the previous-paradigm periodic reporter so the machine has exactly ONE
# reporter (the hook). Everything is BACKED UP (renamed *.bak-tokenfold), never
# hard-deleted, so it is fully reversible. Idempotent: a clean machine prints
# "no legacy reporter found".
cleanup_legacy() {
  local found=0 ts
  ts="$(date +%Y%m%d-%H%M%S 2>/dev/null || echo bak)"

  # 1. macOS launchd agents (any claude-stats / tokenfold push job).
  if [ -d "$HOME/Library/LaunchAgents" ]; then
    for plist in "$HOME/Library/LaunchAgents/"*claude-stats*push*.plist \
                 "$HOME/Library/LaunchAgents/"*tokenfold*push*.plist; do
      [ -f "$plist" ] || continue
      found=1
      local label
      label="$(basename "$plist" .plist)"
      launchctl unload "$plist" 2>/dev/null \
        || launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
      mv "$plist" "$plist.bak-tokenfold-$ts"
      ok "retired launchd agent $label (backup: $(basename "$plist").bak-tokenfold-$ts)"
    done
  fi

  # 2. cron entries (Linux / macOS) running the old pusher.
  if command -v crontab >/dev/null 2>&1; then
    local cur
    cur="$(crontab -l 2>/dev/null || true)"
    if printf '%s\n' "$cur" | grep -qE 'claude-stats-push|usage_push\.py|tokenfold-push'; then
      found=1
      printf '%s\n' "$cur" > "$HOME/.tokenfold-crontab.bak-$ts"
      printf '%s\n' "$cur" | grep -vE 'claude-stats-push|usage_push\.py|tokenfold-push' | crontab -
      ok "removed legacy cron entr(ies) (backup: ~/.tokenfold-crontab.bak-$ts)"
    fi
  fi

  # 3. Stray old pusher scripts + their config (inert once the scheduler is gone).
  for stray in "$HOME/claude-stats-push.py" \
               "$HOOKS_DIR/usage_push.py" \
               "$HOOKS_DIR/config.json"; do
    if [ -f "$stray" ]; then
      found=1
      mv "$stray" "$stray.bak-tokenfold-$ts"
      ok "retired $(basename "$stray") (backup alongside it)"
    fi
  done

  [ "$found" -eq 1 ] || log "no legacy reporter found — nothing to clean up"
  if [ "$found" -eq 1 ]; then
    warn "the old launchd plist embedded STATS_API_KEY in plaintext — consider rotating that token"
  fi
}

if [ "$CLEANUP_ONLY" -eq 1 ]; then
  echo "retiring legacy reporter (backups kept)…"
  cleanup_legacy
  echo; ok "cleanup complete."
  exit 0
fi

# ── Opt-in resident watcher (macOS launchd) ───────────────────────────────────
# Installs the --watch daemon as a LaunchAgent: events land within ~1s instead
# of waiting for the next hook fire. The daemon takes a single-instance flock,
# so the existing Stop/SessionEnd/PostToolUse hooks keep firing harmlessly
# (they just skip while the daemon holds the lock). macOS-only for now.
install_watch_agent() {
  if [ "$(uname -s)" != "Darwin" ]; then
    die "watch-mode install is macOS-only for now (launchd). This host is $(uname -s); the hooks alone still report usage."
  fi
  [ -f "$WATCH_PLIST_SRC" ] || die "missing $WATCH_PLIST_SRC (run from a tokenfold checkout's client/ dir)"
  command -v launchctl >/dev/null 2>&1 || die "launchctl not found — cannot install the watch agent"

  mkdir -p "$LAUNCH_AGENTS"
  local dest="$LAUNCH_AGENTS/$WATCH_LABEL.plist"

  # Path-substitute the placeholders (mirrors bootstrap.sh's __TOKENFOLD_URL__).
  # sed with '|' delimiters so the HOME/hooks paths (which contain '/') are safe.
  sed -e "s|__TOKENFOLD_HOOKS_DIR__|$HOOKS_DIR|g" \
      -e "s|__HOME__|$HOME|g" \
      "$WATCH_PLIST_SRC" > "$dest" || die "could not write $dest"

  # Idempotent (re)load: bootout first (tolerant of "not loaded"), then bootstrap.
  local domain="gui/$(id -u)"
  launchctl bootout "$domain/$WATCH_LABEL" 2>/dev/null || true
  if launchctl bootstrap "$domain" "$dest" 2>/dev/null; then
    ok "resident watcher installed + loaded ($WATCH_LABEL)"
  else
    # Fall back to the legacy verb on older macOS where bootstrap is unavailable.
    launchctl unload "$dest" 2>/dev/null || true
    launchctl load "$dest" 2>/dev/null \
      && ok "resident watcher installed + loaded ($WATCH_LABEL, legacy load)" \
      || die "launchctl could not bootstrap $dest"
  fi
  echo "  logs: ~/.tokenfold-push.log"
}

if [ "$INSTALL_WATCH" -eq 1 ] && [ "$VERIFY_ONLY" -eq 1 ]; then
  die "--watch and --verify-only are mutually exclusive"
fi

# ── Install phase ─────────────────────────────────────────────────────────────
if [ "$VERIFY_ONLY" -eq 0 ]; then
  [ -n "$TOKEN" ] || die "no token. Pass --token <STATS_API_KEY>, set \$TOKENFOLD_API_KEY, or place it in $CONFIG_DIR/tokenfold-api-key"

  # 1. Hook scripts -> ~/.claude/usage-telemetry/
  mkdir -p "$HOOKS_DIR"
  [ -f "$SCRIPT_DIR/claude-stats-push.py" ]   || die "missing $SCRIPT_DIR/claude-stats-push.py (run from a tokenfold checkout's client/ dir)"
  [ -f "$SCRIPT_DIR/tokenfold-usage-push.sh" ] || die "missing $SCRIPT_DIR/tokenfold-usage-push.sh"
  cp "$SCRIPT_DIR/claude-stats-push.py"    "$HOOKS_DIR/tokenfold-push.py"
  cp "$SCRIPT_DIR/tokenfold-usage-push.sh" "$HOOKS_DIR/tokenfold-usage-push.sh"
  chmod +x "$HOOKS_DIR/tokenfold-usage-push.sh"

  # Self-updater (optional file so older checkouts still install cleanly).
  # Once present, the wrapper spawns it after each push and this machine
  # tracks GitHub main without further manual installs.
  if [ -f "$SCRIPT_DIR/tokenfold-update.sh" ]; then
    cp "$SCRIPT_DIR/tokenfold-update.sh" "$HOOKS_DIR/tokenfold-update.sh"
    chmod +x "$HOOKS_DIR/tokenfold-update.sh"
  fi

  # hook.sh wrapper — stamps the machine name, then execs the detaching pusher.
  cat > "$HOOKS_DIR/hook.sh" <<'EOF'
#!/bin/sh
# Stop/SessionEnd hook: flush this machine's new Claude Code events to tokenfold.
# Detaches instantly; reads ~/.config/notify-relay-url + the ingest token.
TOKENFOLD_MACHINE="$(hostname -s)"; export TOKENFOLD_MACHINE
exec "$HOME/.claude/usage-telemetry/tokenfold-usage-push.sh"
EOF
  chmod +x "$HOOKS_DIR/hook.sh"
  ok "hook scripts installed in $HOOKS_DIR"

  # 2. Config: URL (world-readable) + token (0600). Only writes the token we resolved.
  mkdir -p "$CONFIG_DIR"
  printf '%s' "$URL" > "$CONFIG_DIR/notify-relay-url"
  umask 077
  printf '%s' "$TOKEN" > "$CONFIG_DIR/tokenfold-api-key"
  chmod 600 "$CONFIG_DIR/tokenfold-api-key"
  ok "config written ($CONFIG_DIR/notify-relay-url, tokenfold-api-key [0600])"

  # 3. Register the hook as its own group object in Stop + SessionEnd (idempotent).
  [ -f "$SETTINGS" ] && cp "$SETTINGS" "$SETTINGS.bak-tokenfold"
  python3 - "$SETTINGS" <<'PY'
import json, os, sys

path = sys.argv[1]
marker = "usage-telemetry/hook.sh"
group = {"hooks": [{"type": "command",
                    "command": '"$HOME/.claude/usage-telemetry/hook.sh"',
                    "timeout": 10}]}
# Long turns can run for hours before Stop fires; PostToolUse + a 1-min
# debounce (TOKENFOLD_MIN_INTERVAL in the wrapper) keeps data <=1 min stale.
# Stop/SessionEnd stay undebounced — a session ending must always flush.
post_group = {"hooks": [{"type": "command",
              "command": 'TOKENFOLD_MIN_INTERVAL=60 "$HOME/.claude/usage-telemetry/hook.sh"',
              "timeout": 10}]}

try:
    with open(path) as f:
        settings = json.load(f)
except FileNotFoundError:
    settings = {}
except json.JSONDecodeError as e:
    sys.exit(f"settings.json is not valid JSON ({e}); fix it by hand and re-run")

hooks = dict(settings.get("hooks", {}))
changed = False
for event in ("Stop", "SessionEnd", "PostToolUse"):
    groups = list(hooks.get(event, []))
    add = post_group if event == "PostToolUse" else group
    want = add["hooks"][0]["command"]
    present = False
    normalized = []
    for g in groups:
        if not (isinstance(g, dict) and any(
                marker in (h.get("command", "") or "")
                for h in g.get("hooks", []) if isinstance(h, dict))):
            normalized.append(g)       # foreign group — pass through untouched
            continue
        present = True
        # ours: the fleet upgrades by re-running this installer, so a stale
        # command (e.g. an old debounce value) must be rewritten in place
        new_hooks = [
            {**h, "command": want}
            if isinstance(h, dict) and marker in (h.get("command", "") or "")
               and h.get("command") != want
            else h
            for h in g.get("hooks", [])
        ]
        if new_hooks != g.get("hooks", []):
            changed = True
        normalized.append({**g, "hooks": new_hooks})
    groups = normalized
    if not present:
        groups = groups + [add]    # new list — never mutate the original
        changed = True
    if groups != hooks.get(event):
        hooks[event] = groups

if changed:
    new_settings = {**settings, "hooks": hooks}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(new_settings, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    print("REGISTERED")
else:
    print("ALREADY")
PY

  # 4. Retire the legacy periodic reporter (now that the hook is in place).
  if [ "$KEEP_LEGACY" -eq 0 ]; then
    cleanup_legacy
  else
    warn "--keep-legacy: NOT touching any existing launchd/cron reporter (two reporters may run)"
  fi

  # 5. Opt-in resident watcher (macOS launchd). The push script it runs was just
  #    installed to $HOOKS_DIR/tokenfold-push.py above.
  if [ "$INSTALL_WATCH" -eq 1 ]; then
    echo "installing resident watcher (--watch)…"
    install_watch_agent
  fi
fi

# ── Verify phase ──────────────────────────────────────────────────────────────
echo "verifying ingest auth (empty sentinel batch — inserts no events)…"
[ -n "$TOKEN" ] || die "no token available to verify; pass --token"

probe='{"machine":"__install_verify__","project_dir":"__verify__","session_file":"__verify__","cursor":{"last_line_num":0},"events":[]}'
code="$(curl -sS -o /dev/null -w '%{http_code}' \
  -X POST "$URL/api/ingest" \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $TOKEN" \
  --data "$probe" || echo 000)"

case "$code" in
  200) ok "auth OK — server accepted the ingest probe (HTTP 200)" ;;
  401|403) die "auth REJECTED (HTTP $code) — the token does not match the server's STATS_API_KEY" ;;
  000) die "could not reach $URL (network/DNS/TLS). Check the URL and connectivity." ;;
  *)   die "unexpected response (HTTP $code) from $URL/api/ingest" ;;
esac

# ── Real push (proves end-to-end) ─────────────────────────────────────────────
if [ "$VERIFY_ONLY" -eq 0 ] && [ "$DO_PUSH" -eq 1 ]; then
  echo "firing one real push to confirm end-to-end…"
  TOKENFOLD_MACHINE="$(hostname -s)"; export TOKENFOLD_MACHINE
  TOKENFOLD_URL="$URL"; export TOKENFOLD_URL
  TOKENFOLD_API_KEY="$TOKEN"; export TOKENFOLD_API_KEY
  TOKENFOLD_VERBOSE=1; export TOKENFOLD_VERBOSE  # so the pusher reports its accepted/dupe tally
  # Run the pusher in the foreground here (not the detaching wrapper) so we can read its result.
  if python3 "$HOOKS_DIR/tokenfold-push.py" >/tmp/tokenfold-push-verify.log 2>&1; then
    sent="$(grep -Eo 'Done: [0-9]+ accepted, [0-9]+ duplicates' /tmp/tokenfold-push-verify.log | tail -1 || true)"
    [ -n "$sent" ] || sent="no new events since last push"
    ok "push completed — $sent"
  else
    warn "push script exited non-zero — see /tmp/tokenfold-push-verify.log"
    tail -5 /tmp/tokenfold-push-verify.log >&2 || true
  fi
  # Surface any auth failure the pusher logged.
  if grep -qiE 'auth rejected|401|403' /tmp/tokenfold-push-verify.log 2>/dev/null; then
    warn "the push log mentions an auth failure — review /tmp/tokenfold-push-verify.log"
  fi
fi

echo
ok "done. This machine's Stop/SessionEnd hook will flush usage to $URL."
echo "  Account attribution captured from ~/.claude.json oauthAccount (organizationType etc.)."
echo "  To inspect what this machine reports as its account/org, run:"
echo "    python3 -c \"import json,os;o=json.load(open(os.path.expanduser('~/.claude.json'))).get('oauthAccount',{});print({k:o.get(k) for k in ('emailAddress','organizationName','organizationType','organizationUuid')})\""
