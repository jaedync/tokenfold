"""Assemble ledger sessions and their subagent runs from the ledger readers.

A session is a top-level conversation (the orchestrator). Its runs are:
- Claude Code: rows with an agent_id inside the same session (Task subagents).
- Pi: child sessions written by pi-extras, linked by app/pi_lineage.py, and
  any agent_id rows inside the session itself.
A Pi child whose parent cannot be found stands alone: it becomes a session
with no orchestrator cost and itself as its only run, so no dollar is lost.
"""

import sqlite3
import time
from collections import Counter, defaultdict

from .aggregator import _summary_model_identity, canonical_machine
from .ledger_days import billable_threads
from .ledger_query import (activity, billable_groups, group_cost, mixed_scope,
                           project_names, run_tool_counts, titles, tool_counts)
from .pi_lineage import PiLineage, account_class, agent_name
from .pricing import display_model_for_row

PI = "pi-agent"


def _thread() -> dict:
    return {"cost": 0.0, "models": Counter(), "start": None, "end": None,
            "tokens": Counter(), "machine": None, "project_dir": None,
            "client": None, "account": None}


def _span(acc: dict, lo, hi) -> None:
    if lo is not None:
        acc["start"] = lo if acc["start"] is None else min(acc["start"], lo)
    if hi is not None:
        acc["end"] = hi if acc["end"] is None else max(acc["end"], hi)


def collect_threads(rows, enterprise: bool) -> dict:
    """In-scope billable rows -> {(session, agent): thread}."""
    threads: dict = defaultdict(_thread)
    for r in rows:
        t = threads[(r["session_id"], r["agent_id"])]
        cost = group_cost(r, enterprise)
        identity = _summary_model_identity(
            display_model_for_row(r["model"], r["provider"], r["source_client"]))
        t["cost"] += cost
        t["models"][identity] += cost
        t["tokens"].update({"input": r["inp"] or 0, "output": r["outp"] or 0,
                            "cache_read": r["cr"] or 0, "cache_write": r["cc"] or 0})
        _span(t, r["first_ts"], r["last_ts"])
        t["machine"] = t["machine"] or r["machine"]
        t["project_dir"] = t["project_dir"] or r["project_dir"]
        t["client"] = r["source_client"]
        t["account"] = t["account"] or r["org_type"]
    return dict(threads)


class Tree:
    """Root session and parent run for every session in hand."""

    def __init__(self, threads: dict, lineage: PiLineage, force_root: str | None):
        self.threads = threads
        self.lineage = lineage
        agents: dict = defaultdict(list)
        for sid, aid in threads:
            agents[sid].append(aid)
        self.agents = dict(agents)
        self._child: dict = {}
        self.root: dict = {}
        self.parent_run: dict = {}
        for sid in self.agents:
            self._place(sid, force_root)
        members: dict = defaultdict(list)
        for sid in self.agents:
            members[self.root[sid]].append(sid)
        self.members = dict(members)

    def client(self, sid: str) -> str | None:
        aids = self.agents.get(sid)
        return self.threads[(sid, aids[0])]["client"] if aids else None

    def is_child(self, sid: str) -> bool:
        """A Pi session made only of agent rows is a pi-extras or artifact child."""
        if sid not in self._child:
            aids = self.agents.get(sid)
            if aids:
                self._child[sid] = self.client(sid) == PI and None not in aids
            else:
                self._child[sid] = self.lineage.agent_id(sid) is not None
        return self._child[sid]

    def run_id(self, sid: str) -> str | None:
        aids = [a for a in self.agents.get(sid, ()) if a is not None]
        return min(aids) if aids else self.lineage.agent_id(sid)

    def _place(self, sid: str, force_root: str | None) -> None:
        chain: list = []
        if sid != force_root and self.is_child(sid):
            chain = self.lineage.chain(sid)
            if force_root in chain:
                chain = chain[:chain.index(force_root) + 1]
        self.root[sid] = chain[-1] if chain else sid
        near = chain[0] if chain else None
        self.parent_run[sid] = self.run_id(near) if near and self.is_child(near) else None


def _top(models: Counter):
    return max(models, key=models.get) if models else None


def _bounds(*values) -> tuple:
    vals = [v for v in values if v is not None]
    return (min(vals), max(vals)) if vals else (None, None)


def _run(sid, aid, tree, act, tools, run_tools) -> dict:
    thread, a = tree.threads[(sid, aid)], act.get((sid, aid), {})
    start, end = _bounds(thread["start"], thread["end"], a.get("start"), a.get("end"))
    child = tree.is_child(sid)
    pi = thread["client"] == PI
    return {
        "id": aid,
        "parent": tree.parent_run[sid] if child else None,
        "model": _top(thread["models"]),
        "start": start, "end": end,
        "cost_usd": thread["cost"],
        "tools": tools.get(sid, 0) if child else run_tools.get((sid, aid), 0),
        # Claude Code Task descriptions are dropped at ingest (tool input is
        # not stored); a Pi child's name comes from its file name.
        "label": agent_name(aid) if pi else None,
        "active_s": a.get("active_s", 0.0),
        "_models": thread["models"], "_tokens": thread["tokens"],
    }


def _meta(conn: sqlite3.Connection, root: str, tree: Tree) -> dict:
    """Machine/project/client of a root: its own threads, else its first row
    (a parent with no billable rows in the window)."""
    aids = tree.agents.get(root)
    if aids:
        t = tree.threads[(root, aids[0])]
        return {"machine": t["machine"], "project_dir": t["project_dir"],
                "client": t["client"]}
    row = conn.execute("SELECT source_machine, project_dir, source_client FROM events "
                       "WHERE session_id=? LIMIT 1", (root,)).fetchone()
    return {"machine": row[0] if row else None, "project_dir": row[1] if row else None,
            "client": row[2] if row else None}


def _session(conn, root, tree, act, runs, title) -> dict:
    main = tree.threads.get((root, None))
    meta, a = _meta(conn, root, tree), act.get((root, None), {})
    start, end = _bounds(*(([main["start"], main["end"]] if main else [])
                           + [a.get("start"), a.get("end")]
                           + [r["start"] for r in runs] + [r["end"] for r in runs]))
    models = Counter(main["models"]) if main else Counter()
    tokens = Counter(main["tokens"]) if main else Counter()
    for r in runs:
        models.update(r["_models"])
        tokens.update(r["_tokens"])
    orch = main["cost"] if main else 0.0
    agent = sum(r["cost_usd"] for r in runs)
    if title is None and tree.is_child(root) and runs:
        title = runs[0]["label"]
    return {
        "id": root, "machine": canonical_machine(meta["machine"]),
        "project": meta["project_dir"], "title": title, "client": meta["client"],
        "start": start, "end": end,
        "active_s": a.get("active_s", 0.0), "prompts": a.get("prompts", 0),
        "model": _top(main["models"]) if main and main["models"] else _top(models),
        "cost_usd": orch + agent, "orch_cost_usd": orch, "agent_cost_usd": agent,
        # The route sets live from the hook relay state at response time.
        "runs": runs, "live": False,
        "_models": models, "_tokens": tokens,
    }


def assemble(conn, tree: Tree, act: dict, span: tuple) -> list[dict]:
    """Session dicts, internal _models/_tokens keys still attached."""
    tools = tool_counts(conn, sorted(tree.agents), *span)
    inline = sorted({sid for (sid, aid) in tree.threads
                     if aid is not None and not tree.is_child(sid)})
    run_tools = run_tool_counts(conn, inline, *span)
    names = titles(conn, sorted(tree.members))
    out = []
    for root, members in tree.members.items():
        runs = sorted((_run(sid, aid, tree, act, tools, run_tools)
                       for sid in members for aid in tree.agents[sid] if aid is not None),
                      key=lambda r: (r["start"], r["id"]))
        out.append(_session(conn, root, tree, act, runs, names.get(root)))
    return out


def build_sessions(conn, pred: str, scope: str, start: float, end: float,
                   session_ids: list[str] | None = None,
                   force_root: str | None = None) -> list[dict]:
    """Ledger sessions in [start, end). Internal keys start with '_'.

    The window path reuses cached closed days (app/ledger_days.py); the
    detail path (session_ids) scans its few sessions directly.
    """
    def collect(rows):
        return collect_threads(rows, scope == "enterprise")

    if session_ids is None:
        threads = billable_threads(conn, pred, scope, start, end, time.time(), collect)
    else:
        threads = collect(billable_groups(conn, pred, start, end, session_ids))
    known = {}
    for (sid, aid), t in threads.items():
        if t["client"] == PI:
            top = (sid, None) in threads
            known[sid] = (account_class(t["account"]), t["machine"], None if top else aid)
    tree = Tree(threads, PiLineage(conn, known), force_root)
    ids = set(tree.agents) | set(tree.members)
    mixed = mixed_scope(conn, sorted(i for i in ids if not i.startswith("pi:")),
                        start, end, pred)
    act = activity(conn, sorted(ids - mixed), start, end)
    act.update(activity(conn, sorted(ids & mixed), start, end, exact_pred=pred))
    sessions = assemble(conn, tree, act, (start, end))
    names = project_names(s["project"] for s in sessions)
    sessions.sort(key=lambda s: (-(s["end"] or 0), s["id"]))
    return [{**s, "project": names.get(s["project"], s["project"])} for s in sessions]
