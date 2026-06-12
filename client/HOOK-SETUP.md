# Tokenfold Usage Hook — Setup & Verification Runbook

**Point a Claude Code session at this file** to install (or verify) the Stop/SessionEnd
hook that "phones home" this machine's Claude Code usage to your tokenfold server.

You need two inputs from the operator (neither is stored in this repo):
- **Server URL** — the address of your tokenfold dashboard.
- **Ingest token** — the server's `STATS_API_KEY`. Easiest way to retrieve it: log in
  to the dashboard and click **Ingest Key** in the footer (click-to-reveal + copies to
  clipboard). Canonical source: the server's `.env`.

> For an agent: read this whole file, then run the **one command** in step 2. Everything
> else is context and troubleshooting. The installer is idempotent — running it on an
> already-configured machine is a safe no-op plus a re-verification.

---

## What gets installed

| Path | Purpose |
|------|---------|
| `~/.claude/usage-telemetry/tokenfold-push.py` | The pusher: reads new lines from `~/.claude/projects/**/*.jsonl`, batches them, POSTs to `/api/ingest`. Incremental (cursor at `~/.tokenfold-cursor.json`); server-side dedup makes re-runs safe. |
| `~/.claude/usage-telemetry/tokenfold-usage-push.sh` | Detaching wrapper — returns instantly so the hook never blocks your session. |
| `~/.claude/usage-telemetry/hook.sh` | Stamps `TOKENFOLD_MACHINE=$(hostname -s)` and execs the wrapper. This is what the hook calls. |
| `~/.claude/usage-telemetry/tokenfold-update.sh` | **Self-updater.** Spawned detached after each debounced push: compares the latest GitHub commit touching `client/` against a local stamp (`.client-sha`) and, on change, downloads that exact commit's tarball and re-runs the downloaded installer. After this lands once, the machine tracks `main` with **no further manual installs**. Logs to `update.log`; disable with `TOKENFOLD_NO_UPDATE=1`. |
| `~/.config/notify-relay-url` | Server base URL. |
| `~/.config/tokenfold-api-key` (mode `0600`) | The ingest token (`STATS_API_KEY`). **Never committed to git.** |
| `~/.claude/settings.json` → `hooks.Stop`, `hooks.SessionEnd` and `hooks.PostToolUse` | A new hook **group object** calling `hook.sh`. Appended, never merged into an existing group, so it can't disturb your other hooks. |

**Account attribution** is automatic: each push reads `~/.claude.json`'s `oauthAccount`
(email, `organizationName`, `organizationType`, `organizationUuid`) and tags every event.
Personal Max accounts report `organizationType=claude_max` → classified **personal**.
Enterprise/Team orgs report `claude_enterprise`/`claude_team` → classified **enterprise**.
**Freshness:** `Stop`/`SessionEnd` flush when a turn or session ends (always, undebounced), but a single turn can run for hours — so the `PostToolUse` hook also fires (after every tool call) with a built-in **1-minute debounce** (`TOKENFOLD_MIN_INTERVAL=60`): data is never more than ~1 minute stale during long runs, at most one push per minute.

No secrets leave the machine — only those four identity fields.

**Auto-update:** pushing to `main` on GitHub is the release channel. Active
machines pick up client changes within one push cycle (the updater checks the
GitHub API at most once per debounced push, ≤12/hr, and no-ops in under a
second when nothing changed). The stamp only advances when the re-run
installer exits 0, so a failed update retries on the next push while the
existing scripts keep working. Machines installed before the updater existed
need **one** manual re-run of the installer to start tracking.

---

## This REPLACES the old periodic reporter

The previous paradigm pushed usage on a **timer** — a macOS **launchd** agent
(`claude-stats-push`, every 5 min) or a **cron** job running
`claude-stats-push.py` / `usage_push.py`. The new hook is **event-driven** (fires on
Stop/SessionEnd, instant) and captures `org_type`/`org_uuid` for correct
enterprise-vs-personal attribution, which the old pusher did not.

You do **not** want both running on one machine: it's two reporters, and if the old
(attribution-blind) one wins a push race, enterprise events land **unclassified →
personal**. So the installer **retires the legacy reporter by default** as its last
install step — only after the new hook is in place. Everything is **backed up**
(renamed `*.bak-tokenfold-<timestamp>`), never hard-deleted, so it's fully reversible:

- launchd: `launchctl unload`ed, plist → `~/Library/LaunchAgents/…plist.bak-tokenfold-*`
- cron: matching lines stripped, full crontab saved to `~/.tokenfold-crontab.bak-*`
- stray scripts: `~/claude-stats-push.py`, `~/.claude/usage-telemetry/usage_push.py`,
  old `config.json` → `*.bak-tokenfold-*`

Flags:
- `--keep-legacy` — install the hook but leave the old reporter alone (NOT recommended;
  you'll have two reporters).
- `--cleanup-only` — just retire the legacy reporter on an already-migrated machine,
  without re-installing.

> ⚠️ The old launchd plist stored `STATS_API_KEY` in **plaintext**. After migrating your
> machines, **rotate that token** (change `STATS_API_KEY` on the server, re-issue to clients).

> **Other clients are unaffected until you migrate them.** A machine still on the old
> launchd/cron keeps reporting as before. Migrate each machine by running the installer on
> it — there's no flag-day cutover.

---

## 1. Prerequisites

- `python3` and `curl` on `PATH` (both standard on macOS/Linux).
- A **tokenfold checkout** so the source scripts are available. If you don't have one:
  ```bash
  git clone <tokenfold-repo> ~/tokenfold && cd ~/tokenfold/client
  ```
  (Or just `cd` into the `client/` directory of an existing checkout.)
- The **ingest token** = the server's `STATS_API_KEY`. Get it from the operator
  (Jaedyn). **Do not paste it into any file that gets committed.** Pass it as a
  command-line argument or export it as `$TOKENFOLD_API_KEY`.

---

## 2. Install (the one command)

From the `client/` directory of a tokenfold checkout:

```bash
./install-tokenfold-hook.sh --url '<SERVER_URL>' --token '<STATS_API_KEY>'
```

That single command:
1. Copies the hook scripts into `~/.claude/usage-telemetry/`.
2. Writes `~/.config/notify-relay-url` + `~/.config/tokenfold-api-key` (token file `0600`).
3. Registers the hook in `hooks.Stop` **and** `hooks.SessionEnd` (idempotent — backs up
   `settings.json` to `settings.json.bak-tokenfold` first).
4. **Retires the legacy launchd/cron reporter** (backups kept; see section above).
5. **Verifies auth** with an empty sentinel batch (inserts no events) → expects HTTP `200`.
6. **Fires one real push** and reports `Done: N accepted, M duplicates`.

**Success looks like:**
```
  ✓ hook scripts installed in /Users/you/.claude/usage-telemetry
  ✓ config written (…/notify-relay-url, tokenfold-api-key [0600])
REGISTERED                       # or ALREADY on a re-run
  ✓ retired launchd agent claude-stats-push (backup: …bak-tokenfold-…)
  ✓ auth OK — server accepted the ingest probe (HTTP 200)
  ✓ push completed — Done: 16 accepted, 0 duplicates
  ✓ done. This machine's Stop/SessionEnd hook will flush usage to <your server URL>.
```

If the token is already on the machine (from a prior install or a notify hook), you can
omit `--token` — it auto-resolves from `$TOKENFOLD_API_KEY`, then
`~/.config/tokenfold-api-key`, then `~/.config/notify-relay-token`.

---

## 3. Verify an existing install (no changes)

```bash
./install-tokenfold-hook.sh --verify-only
```

Runs only the auth probe against the current config. Use this to confirm a machine is
still wired up without touching any files.

---

## 4. Confirm it landed on the dashboard

1. The push tally (`Done: N accepted`) already proves events reached the server.
2. Optionally, check what this machine reports as its identity:
   ```bash
   python3 -c "import json,os;o=json.load(open(os.path.expanduser('~/.claude.json'))).get('oauthAccount',{});print({k:o.get(k) for k in ('emailAddress','organizationName','organizationType','organizationUuid')})"
   ```
3. Open the dashboard (HTTP Basic auth) and confirm
   this machine appears with fresh activity. Personal usage shows under the **personal**
   scope; enterprise orgs under **enterprise**.

---

## 5. Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| `auth REJECTED (HTTP 401/403)` | Token ≠ server's `STATS_API_KEY`. Re-run with the correct `--token`. |
| `could not reach … (HTTP 000)` | Network/DNS/TLS. Check the URL and connectivity; the server uses Caddy + a valid cert. |
| `settings.json is not valid JSON` | Your `settings.json` is malformed. Fix it by hand and re-run; the installer won't overwrite broken JSON. |
| Hook installed but no new events | Cursor (`~/.tokenfold-cursor.json`) may already be current — that's normal; `push completed — no new events` means nothing new since last flush. New activity flushes on the next Stop/SessionEnd. |
| Push log shows `Token refresh failed: 403` / `Usage fetch 401` | Only the **OAuth usage gauges** are affected (the personal weekly/5h limit display), **not** event/cost ingestion. Re-auth `~/.claude/.credentials.json` by running `claude` and logging in. |
| Want detailed push output | `TOKENFOLD_VERBOSE=1 python3 ~/.claude/usage-telemetry/tokenfold-push.py` |

**Rollback:** the previous `settings.json` is saved as
`~/.claude/settings.json.bak-tokenfold`. The hook does nothing destructive — to disable it,
remove the group object that calls `usage-telemetry/hook.sh` from `hooks.Stop` /
`hooks.SessionEnd`. To restore the old launchd reporter, rename its
`…plist.bak-tokenfold-*` back and `launchctl load` it (but don't run both reporters).

---

## How the hook fits Claude Code

Claude Code runs `hooks.Stop` when a session turn ends and `hooks.SessionEnd` when a
session closes. Each is a **list of independent group objects**; the installer appends its
own group rather than editing yours, so it coexists with notify/activity-light/debug hooks.
The wrapper detaches the pusher (`setsid`/`nohup`) so your session never waits on the
network. Pushes are incremental and the server deduplicates, so a missed or doubled Stop
event never loses or double-counts data.
