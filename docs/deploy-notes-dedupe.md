# Deploy notes: Meridian/Pi dedupe

## Rollout

- The first start runs a one-time migration and a full rollup rebuild. Both run
  at startup, before the service serves any request. Expect downtime for the
  length of the rebuild (the log shows "rebuilding daily rollups" and
  "daily rollups rebuilt in N s").
- The migration records the version in `meta` (`meridian_pi_dedupe_version`)
  and fills `pi_mirror_sessions`. Later starts skip it.

## Rollback to main

- `main` has no version check. After a rollback, the deduplicated
  `daily_summary` rows stay in place while live window queries (`/api/ha`,
  `/api/rate-limits`) count Meridian-routed requests twice again. The two
  views then disagree.
- To roll back safely: run `DELETE FROM daily_summary` and restart the
  service. `main` then rebuilds the rollups from raw events.
- `pi_mirror_sessions` and the new indexes are harmless to `main`. Leave them.

## Metric changes

- Enterprise (and personal) session and prompt counts change slightly. A
  direct Claude Code session with no real model reply (slash commands only, an
  interrupt before the reply, API errors only) no longer counts as a session
  or a prompt. Cost is unchanged.
- Meridian SDK twin sessions no longer count in sessions, prompts, active
  time, tool calls, agent runs, or machine last-active.
