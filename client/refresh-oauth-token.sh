#!/usr/bin/env bash
# refresh-oauth-token.sh: retired. It does nothing now.
#
# This script refreshed the Claude login in ~/.claude/.credentials.json. A
# refresh rotates the token on Anthropic's side, so Claude Code and Meridian
# on the same machine then held a spent token. Tokenfold now only reads the
# file: Claude Code renews it when it runs.
#
# The file stays so that an old cron line runs without an error. Remove the
# cron line: run `crontab -e` and delete the line with refresh-oauth-token.sh.

echo "[$(date '+%Y-%m-%d %H:%M:%S')] refresh-oauth-token.sh is retired and does nothing. Remove its cron line (crontab -e)."
exit 0
