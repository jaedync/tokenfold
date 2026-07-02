#!/usr/bin/env bash
# Smoke-check /api/ha. Exits non-zero if any required key is missing or malformed.
#
# Usage: scripts/ha-smoke.sh [base_url]
#   base_url defaults to http://localhost:5000
set -euo pipefail

BASE="${1:-http://localhost:5000}"
URL="$BASE/api/ha"

body=$(curl -sf "$URL") || { echo "FAIL: could not fetch $URL"; exit 1; }

check() {
  local expr="$1" desc="$2"
  if ! echo "$body" | jq -e "$expr" >/dev/null; then
    echo "FAIL: $desc"
    echo "Response was:"
    echo "$body" | jq .
    exit 1
  fi
}

check '.cost_today_usd | type == "number"' 'cost_today_usd must be a number'
check '.cost_total_usd | type == "number"' 'cost_total_usd must be a number'
check 'has("five_hour")'                    'five_hour key must be present'
check 'has("weekly")'                       'weekly key must be present'
check 'has("updated_at_epoch")'             'updated_at_epoch key must be present'

# If the window blocks are present, their sub-shape must be complete.
for win in five_hour weekly; do
  if [ "$(echo "$body" | jq ".$win")" != "null" ]; then
    for sub in pct_used spend_usd implied_limit_usd resets_at resets_in_s; do
      check ".$win | has(\"$sub\")" "$win.$sub must be present when $win is not null"
    done
    check ".$win.resets_at | endswith(\":00+00:00\")" \
      "$win.resets_at must be minute-truncated"
  fi
done

# model_buckets: key always present; when populated, each per-model entry
# must carry the full sub-shape with a minute-truncated (or null) resets_at.
check 'has("model_buckets")'                'model_buckets key must be present'
if [ "$(echo "$body" | jq '.model_buckets')" != "null" ]; then
  check '.model_buckets | type == "object"' 'model_buckets must be an object'
  for slug in $(echo "$body" | jq -r '.model_buckets | keys[]'); do
    for sub in pct_used resets_at resets_in_s; do
      check ".model_buckets[\"$slug\"] | has(\"$sub\")" \
        "model_buckets.$slug.$sub must be present"
    done
    check ".model_buckets[\"$slug\"].resets_at | (. == null) or endswith(\":00+00:00\")" \
      "model_buckets.$slug.resets_at must be minute-truncated or null"
  done
fi

# updated_at_epoch, if present, must be divisible by 10.
if [ "$(echo "$body" | jq '.updated_at_epoch')" != "null" ]; then
  check '.updated_at_epoch % 10 == 0' 'updated_at_epoch must be divisible by 10'
fi

echo "OK: /api/ha shape is valid"
