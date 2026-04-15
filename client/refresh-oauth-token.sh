#!/usr/bin/env bash
# refresh-oauth-token.sh — Keep Claude OAuth token fresh on the host.
#
# Reads ~/.claude/.credentials.json, checks if the access token is expiring
# within 30 minutes, and refreshes it via the OAuth token endpoint directly.
# Tokenfold's usage_fetcher picks up the fresher token from the mounted file.
#
# Install as a cron job (every 30 minutes):
#   crontab -e
#   */30 * * * * /path/to/tokenfold/client/refresh-oauth-token.sh >> /tmp/claude-token-refresh.log 2>&1

set -euo pipefail

CREDS_FILE="${HOME}/.claude/.credentials.json"
TOKEN_URL="https://platform.claude.com/v1/oauth/token"
CLIENT_ID="9d1c250a-e61b-44d9-88ed-5944d1962f5e"
SCOPES="user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"
# Refresh if less than 30 minutes remaining (in milliseconds)
REFRESH_THRESHOLD_MS=1800000

ts() { date '+%Y-%m-%d %H:%M:%S'; }

if [ ! -f "$CREDS_FILE" ]; then
    echo "[$(ts)] No credentials file at $CREDS_FILE"
    exit 0
fi

# Use Python (stdlib only) to check expiry and refresh if needed.
# This mirrors the logic in claude-stats-push.py's _get_oauth_token().
python3 - "$CREDS_FILE" "$TOKEN_URL" "$CLIENT_ID" "$SCOPES" "$REFRESH_THRESHOLD_MS" <<'PYEOF'
import json, sys, time, urllib.request, os

creds_path = sys.argv[1]
token_url = sys.argv[2]
client_id = sys.argv[3]
scopes = sys.argv[4]
threshold_ms = int(sys.argv[5])

try:
    with open(creds_path) as f:
        creds = json.load(f)
except (json.JSONDecodeError, OSError) as e:
    print(f"Cannot read credentials: {e}")
    sys.exit(1)

oauth = creds.get("claudeAiOauth", {})
token = oauth.get("accessToken")
if not token:
    print("No access token in credentials file")
    sys.exit(0)

expires_at = oauth.get("expiresAt", 0)
now_ms = time.time() * 1000
remaining_min = (expires_at - now_ms) / 1000 / 60

if (expires_at - now_ms) > threshold_ms:
    print(f"Token valid for {remaining_min:.0f} more minutes, no refresh needed")
    sys.exit(0)

refresh_token = oauth.get("refreshToken")
if not refresh_token:
    print("No refresh token available — manual re-auth required")
    sys.exit(1)

print(f"Token expiring in {remaining_min:.0f}m, refreshing...")

body = json.dumps({
    "grant_type": "refresh_token",
    "refresh_token": refresh_token,
    "client_id": client_id,
    "scope": scopes,
}).encode()

req = urllib.request.Request(
    token_url,
    data=body,
    headers={
        "Content-Type": "application/json",
    },
    method="POST",
)

try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
except Exception as e:
    print(f"Token refresh request failed: {e}")
    sys.exit(1)

new_token = data.get("access_token")
if not new_token:
    print("Refresh response missing access_token")
    sys.exit(1)

oauth["accessToken"] = new_token
oauth["refreshToken"] = data.get("refresh_token", refresh_token)
oauth["expiresAt"] = int(time.time() * 1000) + data.get("expires_in", 7200) * 1000
creds["claudeAiOauth"] = oauth

# Atomic write: write to tmp file then rename
tmp_path = creds_path + ".tmp"
try:
    with open(tmp_path, "w") as f:
        json.dump(creds, f, indent=2)
    os.rename(tmp_path, creds_path)
except OSError as e:
    print(f"Failed to write credentials: {e}")
    # Clean up tmp file if rename failed
    try:
        os.unlink(tmp_path)
    except OSError:
        pass
    sys.exit(1)

new_remaining = (oauth["expiresAt"] - time.time() * 1000) / 1000 / 60
print(f"Token refreshed, valid for {new_remaining:.0f} more minutes")
PYEOF
