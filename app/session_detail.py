"""GET /api/sessions/{id}: one ledger session over its whole life, plus detail.

The session is built by the same code as /api/ledger (no time clip), so its
cost, runs and model names match the ledger row. Pi child sessions are found
through app/pi_lineage.py: children run on the parent's machine, so the
candidates are that machine's Pi agent rows during the parent's life.
"""

import json
from collections import Counter

from .aggregator import _conditional_model_names, display_tool_name
from .config import scope_predicate
from .event_dedupe import not_pi_copy
from .ledger import public_session
from .ledger_build import build_sessions
from .pi_lineage import PiLineage, account_class, split_pi_id

ALL_TIME = (0.0, 1e12)
# A child can outlive the parent's last event (the parent may exit first).
CHILD_GRACE_S = 86400
# Write tools by their bare name (MCP prefixes stripped): Claude Code, Pi, MCP.
EDIT_TOOLS = {"edit", "write", "multiedit", "notebookedit", "apply_patch"}
_IDS = "(SELECT value FROM json_each(?))"


def served_state(rows) -> dict:
    """(requested model, served model, blocks) rows -> session state.

    as_asked: every signed block names the requested model. rerouted: some
    block names another model. partly_hidden: some header names no model
    (v4 signatures), the rest as asked. unknown: no signed block at all.
    """
    blocks = sum(n for _m, _s, n in rows)
    if not blocks:
        return {"state": "unknown", "blocks": 0}
    if any(s is not None and s != m for m, s, _n in rows):
        state = "rerouted"
    elif any(s is None for _m, s, _n in rows):
        state = "partly_hidden"
    else:
        state = "as_asked"
    return {"state": state, "blocks": blocks}


def _descendants(conn, sid: str) -> list[str]:
    """Pi child sessions (any depth) of Pi session `sid`."""
    if split_pi_id(sid) is None:
        return []
    head = conn.execute(
        "SELECT MIN(ts_epoch), MAX(ts_epoch), MAX(source_machine), MAX(org_type) "
        "FROM events WHERE session_id=?", (sid,)).fetchone()
    if head is None or head[0] is None:
        return []
    rows = conn.execute(
        "SELECT DISTINCT session_id, agent_id, source_machine, org_type FROM events "
        "INDEXED BY idx_events_billable_ts WHERE type='assistant' AND request_id IS NOT NULL "
        "AND ts_epoch >= ? AND ts_epoch < ? AND source_client='pi-agent' "
        "AND agent_id IS NOT NULL AND source_machine=? AND org_type=?",
        (head[0], head[1] + CHILD_GRACE_S, head[2], head[3])).fetchall()
    known = {r[0]: (account_class(r[3]), r[2], r[1]) for r in rows}
    lineage = PiLineage(conn, known)
    return sorted(c for c in known if c != sid and sid in lineage.chain(c))


def _tools(conn, ids: list[str]) -> list[dict]:
    calls: Counter = Counter()
    for name, n in conn.execute(
            f"SELECT name, COUNT(*) FROM tool_uses WHERE session_id IN {_IDS} "
            f"AND {not_pi_copy('tool_uses')} GROUP BY name",
            (json.dumps(ids),)):
        calls[display_tool_name(name)] += n
    # tool_uses.is_error is never set by ingest (no tool_result linkage), so
    # an error count would be a made-up zero.
    return [{"name": n, "calls": c, "errors": None} for n, c in calls.most_common()]


def _file_edits(conn, ids: list[str]) -> int:
    n = 0
    for name, count in conn.execute(
            f"SELECT name, COUNT(*) FROM tool_uses WHERE session_id IN {_IDS} "
            f"AND {not_pi_copy('tool_uses')} GROUP BY name",
            (json.dumps(ids),)):
        if name.rsplit("__", 1)[-1].lower() in EDIT_TOOLS:
            n += count
    return n


def _prompt_times(conn, sid: str, pred: str) -> list[float]:
    return [r[0] for r in conn.execute(
        "SELECT ts_epoch FROM events WHERE session_id=? AND type='user' "
        f"AND is_human_prompt=1 AND agent_id IS NULL AND {not_pi_copy()} AND {pred} "
        "ORDER BY ts_epoch", (sid,))]


def _served(conn, ids: list[str], scope: str, pred: str) -> dict:
    # Served-model capture is a personal-scope feature (/api/served-models
    # 404s for enterprise); do not surface it through this route either.
    if scope != "personal":
        return {"state": "unknown", "blocks": 0}
    rows = conn.execute(
        "SELECT model, served_model, COUNT(*) FROM events "
        f"WHERE session_id IN {_IDS} AND sig_header IS NOT NULL AND model IS NOT NULL "
        f"AND {pred} GROUP BY model, served_model", (json.dumps(ids),)).fetchall()
    return served_state([(r[0], r[1], r[2]) for r in rows])


def build_detail(conn, scope: str, sid: str, live=frozenset()) -> dict | None:
    pred = scope_predicate(scope)
    ids = [sid, *_descendants(conn, sid)]
    sessions = build_sessions(conn, pred, scope, *ALL_TIME, session_ids=ids,
                              force_root=sid)
    found = next((s for s in sessions if s["id"] == sid), None)
    if found is None:
        return None
    members = sorted(ids)
    names = _conditional_model_names(set(found["_models"]))
    total = found["cost_usd"]
    out = public_session(found, names, live)
    out.update({
        "prompt_times": _prompt_times(conn, sid, pred),
        "models": [{"model": names[i], "cost_usd": usd,
                    "share": usd / total if total else None}
                   for i, usd in found["_models"].most_common()],
        "tools": _tools(conn, members),
        "tokens": {k: found["_tokens"].get(k, 0)
                   for k in ("input", "output", "cache_read", "cache_write")},
        "file_edits": _file_edits(conn, members),
        "served_model": _served(conn, members, scope, pred),
    })
    return out
