"""Canonical event filters shared by live and rolled-up usage accounting."""

import hashlib
import logging
import re
import sqlite3
from datetime import date, timedelta
from .config import enterprise_predicate

logger = logging.getLogger(__name__)

_MESSAGE_ID_BATCH = 500
_COMPONENTS = ('input', 'output', 'cache_read', 'cache_write')


def _positive_reported_cost(alias: str) -> str:
    """Pi cost evidence: pricing.reported_cost's total-wins rule, positive only.

    Anthropic never serves tokens for free, so an explicit $0 means Pi had no
    rates for the model (production: claude-fable-5-1 rows on 2026-09-02,
    3.7M tokens). Such a row yields to its priced SDK twin like a row with
    no reported cost at all.
    """
    parts = ' + '.join(f'COALESCE({alias}.reported_cost_{f},0)' for f in _COMPONENTS)
    return f'(COALESCE({alias}.reported_cost_total, {parts}) > 0)'


# ── Pi copies (cost and activity) ──
# A Pi session file copied to another machine, or moved to another path, is
# reported again under a new namespace (ingest._pi_namespace hashes the
# machine and the session file), so its uuids, session ids and request ids
# all differ, and so do the client's event ids when the path changed. A Pi
# /fork writes the history into a new native session. The provider response
# ID and each entry's content stay the same (production 2026-10: 19,484
# duplicated requests, identical timestamps, tokens and reported costs).
#
# pi_copy_key identifies one Pi entry independent of where it was reported.
# The first-ingested row of a key is the original (in production the machine
# that ran the session, in every cluster but one row). A row is a copy when
# an earlier row in another session of the same work/personal class has its
# key. The flag is stored at ingest, so an arriving copy never changes the
# original or any earlier rollup. Copies are excluded from cost and activity.

_PI_NAMESPACE = re.compile(r'pi:[0-9a-f]{40}:')
_KEY_BACKFILL_ROWS = 50_000
_PI_ENTRY_FIELDS = (
    'type', 'ts_epoch', 'is_human_prompt', 'is_sidechain', 'text_length',
    'thinking_length', 'has_text', 'has_thinking', 'has_tool_use',
    'has_tool_result', 'has_image', 'model', 'provider', 'usage_kind',
    'input_tokens', 'output_tokens', 'cache_read_tokens',
    'cache_creation_tokens', 'reasoning_tokens', 'stop_reason')


def _native(value):
    """The client's own ID inside a ``pi:<40 hex>:`` namespaced value."""
    if isinstance(value, str) and _PI_NAMESPACE.match(value):
        return value[44:]
    return value


def pi_copy_key(row) -> str | None:
    """Location-independent identity of a Pi row (an events row mapping).

    An Anthropic msg_* ID is globally unique, so it alone names the request.
    Any other row is named by its content: timestamp, flags, sizes, model
    and tokens, plus a real provider response ID (an 'event:' fallback ID
    hashes the file path, so it is left out). Subagent rows also keep their
    native session: equal content in two child sessions is not proof of one
    request (production has a few such rows, none with a shared response ID).
    """
    if row['source_client'] != 'pi-agent':
        return None
    message_id = row['message_id']
    if (row['type'] == 'assistant' and isinstance(message_id, str)
            and message_id.startswith('msg_') and len(message_id) > 4):
        basis = 'msg\0' + message_id
    else:
        response = _native(row['request_id']) or ''
        if response.startswith('event:'):
            response = ''
        session = (_native(row['session_id']) or '') if row['is_sidechain'] else ''
        basis = '\0'.join(['entry', session, response]
                          + [repr(row[f]) for f in _PI_ENTRY_FIELDS])
    return hashlib.sha256(basis.encode('utf-8', 'surrogatepass')).hexdigest()[:32]


def not_pi_copy(alias: str = 'events') -> str:
    """Row filter for events or tool_uses. Aliases are internal trusted names."""
    return f'COALESCE({alias}.pi_copy,0)=0'


def pi_copy_flag_sql(target: str) -> str:
    """UPDATE that flags the copies among the rows matching ``target``.

    ``target`` is trusted SQL over the outer events row. INDEXED BY pins the
    partial key index: the planner must not walk the rowid range instead.
    """
    return (
        f"UPDATE events SET pi_copy=1 WHERE {target} AND pi_copy_key IS NOT NULL "
        "AND COALESCE(pi_copy,0)=0 AND EXISTS (SELECT 1 FROM events AS first_seen "
        "INDEXED BY idx_events_pi_copy_key "
        "WHERE first_seen.pi_copy_key=events.pi_copy_key "
        "AND first_seen.rowid<events.rowid "
        "AND first_seen.session_id IS NOT events.session_id "
        f"AND {enterprise_predicate('first_seen')}={enterprise_predicate('events')})")


def flag_new_pi_copies(conn: sqlite3.Connection, events_after: int,
                       tools_after: int) -> int:
    """Flag copies among rows inserted after the given rowids (caller owns
    the write transaction, so the order is the ingest order)."""
    flagged = conn.execute(pi_copy_flag_sql('events.rowid > ?'),
                           (events_after,)).rowcount
    if flagged:
        conn.execute(
            "UPDATE tool_uses SET pi_copy=1 WHERE rowid > ? AND event_uuid IN "
            "(SELECT uuid FROM events WHERE rowid > ? AND pi_copy=1)",
            (tools_after, events_after))
    return flagged


# Pi rows with no key: written by code without the key (the first start after
# the upgrade, or a rollback). The partial index idx_events_pi_unkeyed (db.py)
# holds only these rows, so the check on each start costs nothing otherwise.
_UNKEYED_PI = "source_client='pi-agent' AND pi_copy_key IS NULL"


def mark_pi_copies(conn: sqlite3.Connection) -> set[str]:
    """Key the unkeyed Pi rows and flag copies among all rows of their keys.

    Runs on each start; the caller owns the transaction. All rows of a new
    key are checked, so a row keyed here can also be the original of a row
    keyed earlier. Returns the days whose rollups change: the days of the
    newly flagged rows, and each next day, whose active time is seeded from
    the last activity row of the day before (summarizer Q5).
    """
    columns = {r[1] for r in conn.execute('PRAGMA table_info(events)')}
    if not columns.issuperset(('session_id', 'request_id', *_PI_ENTRY_FIELDS)):
        return set()  # a legacy table from before Pi ingest holds no Pi rows
    if conn.execute(f'SELECT 1 FROM events WHERE {_UNKEYED_PI} LIMIT 1').fetchone() is None:
        return set()
    select = (
        "SELECT rowid, source_client, message_id, request_id, session_id, "
        f"{', '.join(_PI_ENTRY_FIELDS)} FROM events "
        f"WHERE {_UNKEYED_PI} AND rowid > ? ORDER BY rowid LIMIT ?")
    conn.execute('CREATE TEMP TABLE pi_copy_new_keys(key TEXT PRIMARY KEY)')
    try:
        # Pages keep memory bounded (production: 515k Pi rows of 2.9M).
        last = 0
        while rows := conn.execute(select, (last, _KEY_BACKFILL_ROWS)).fetchall():
            keys = [(pi_copy_key(r), r['rowid']) for r in rows]
            conn.executemany('UPDATE events SET pi_copy_key=? WHERE rowid=?', keys)
            conn.executemany('INSERT OR IGNORE INTO temp.pi_copy_new_keys VALUES(?)',
                             [(key,) for key, _ in keys])
            last = rows[-1]['rowid']
        days = {r[0] for r in conn.execute(pi_copy_flag_sql(
            'events.pi_copy_key IN (SELECT key FROM temp.pi_copy_new_keys)')
            + ' RETURNING day')}
    finally:
        conn.execute('DROP TABLE temp.pi_copy_new_keys')
    if days:
        conn.execute(
            "UPDATE tool_uses SET pi_copy=1 WHERE source_client='pi-agent' "
            "AND COALESCE(pi_copy,0)=0 AND event_uuid IN "
            "(SELECT uuid FROM events WHERE pi_copy=1)")
    return days | {(date.fromisoformat(d) + timedelta(days=1)).isoformat() for d in days}


def _billable(alias: str) -> str:
    return (f"{alias}.type='assistant' AND {alias}.request_id IS NOT NULL "
            f"AND {alias}.model IS NOT NULL AND {alias}.model!='<synthetic>'")


def _same_response(left: str, right: str) -> str:
    return (f"{left}.message_id={right}.message_id "
            f"AND {enterprise_predicate(left)}={enterprise_predicate(right)}")


def canonical_assistant_predicate(alias: str = 'events') -> str:
    """Prefer a positive reported Pi cost; otherwise retain the paid SDK fallback.

    Only exact Anthropic msg_* identities in the same configured work/personal
    class match. Synthetic Pi account identities prevent email equality joins.
    Missing usage or an explicit $0 is not cost evidence. Such a Pi row yields
    to its matching SDK row (with tokens) so the request is billed once.
    Without a match, existing standalone source pricing is unchanged. Aliases
    are internal trusted names.
    """
    sdk_has_cost_twin = (
        "EXISTS (SELECT 1 FROM events AS canonical_pi WHERE "
        f"{_billable('canonical_pi')} AND canonical_pi.source_client='pi-agent' "
        "AND LOWER(canonical_pi.provider)='anthropic' "
        f"AND {not_pi_copy('canonical_pi')} "
        f"AND {_positive_reported_cost('canonical_pi')} "
        f"AND {_same_response('canonical_pi', alias)})"
    )
    # An SDK row with no tokens and no server-tool use adds no cost, so it
    # yields to any billable Pi row of the response: otherwise both survive
    # and api_calls counts the request twice. An SDK row with a blank or
    # '<unknown>' model but real tokens is kept, since it may be the only
    # cost evidence when Pi has none (a known api_calls double count).
    sdk_is_empty_copy = (
        f"(COALESCE({alias}.input_tokens,0) + COALESCE({alias}.output_tokens,0) "
        f"+ COALESCE({alias}.cache_creation_tokens,0) + COALESCE({alias}.cache_read_tokens,0)) = 0 "
        f"AND COALESCE({alias}.web_search_requests,0) = 0 "
        "AND EXISTS (SELECT 1 FROM events AS empty_pi WHERE "
        f"{_billable('empty_pi')} AND empty_pi.source_client='pi-agent' "
        "AND LOWER(empty_pi.provider)='anthropic' "
        f"AND {not_pi_copy('empty_pi')} "
        f"AND {_same_response('empty_pi', alias)})"
    )
    sdk = (
        f"{alias}.source_client='claude-code' AND {alias}.message_id GLOB 'msg_?*' "
        f"AND ({alias}.provider IS NULL OR LOWER({alias}.provider)='anthropic') "
        f"AND ({sdk_has_cost_twin} OR ({sdk_is_empty_copy}))"
    )
    pi = (
        f"{alias}.source_client='pi-agent' AND LOWER({alias}.provider)='anthropic' "
        f"AND {alias}.message_id GLOB 'msg_?*' AND NOT {_positive_reported_cost(alias)} "
        "AND EXISTS (SELECT 1 FROM events AS fallback_sdk WHERE "
        f"{_billable('fallback_sdk')} AND fallback_sdk.source_client='claude-code' "
        "AND (fallback_sdk.provider IS NULL OR LOWER(fallback_sdk.provider)='anthropic') "
        "AND fallback_sdk.model NOT IN ('', '<unknown>') "
        "AND (COALESCE(fallback_sdk.input_tokens,0) + COALESCE(fallback_sdk.output_tokens,0) "
        "+ COALESCE(fallback_sdk.cache_creation_tokens,0) "
        "+ COALESCE(fallback_sdk.cache_read_tokens,0)) > 0 "
        f"AND {_same_response('fallback_sdk', alias)})"
    )
    # Twin lookups see only original Pi rows, so a copy with different cost
    # evidence cannot make the SDK row and the original both yield.
    return f'{not_pi_copy(alias)} AND NOT ({sdk}) AND NOT ({pi})'


def _matching_days(conn: sqlite3.Connection, message_ids: set[str], source: str) -> set[str]:
    ids = sorted(message_id for message_id in message_ids if message_id)
    # State the partial-index condition explicitly for the Pi-side reroll scan.
    provider_filter = "AND LOWER(provider)='anthropic' " if source == 'pi-agent' else ''
    days: set[str] = set()
    for offset in range(0, len(ids), _MESSAGE_ID_BATCH):
        batch = ids[offset:offset + _MESSAGE_ID_BATCH]
        placeholders = ','.join('?' for _ in batch)
        rows = conn.execute(
            "SELECT DISTINCT day FROM events WHERE type='assistant' AND source_client=? "
            f"{provider_filter}AND message_id IN ({placeholders}) "
            f"AND NOT ({canonical_assistant_predicate()})",
            [source, *batch],
        )
        days.update(row['day'] for row in rows)
    return days


def matching_claude_days(conn: sqlite3.Connection, message_ids: set[str]) -> set[str]:
    """SDK days invalidated by newly arrived authoritative Pi costs."""
    return _matching_days(conn, message_ids, 'claude-code')


def matching_pi_days(conn: sqlite3.Connection, message_ids: set[str]) -> set[str]:
    """Pi days without positive reported cost, invalidated by a newly arrived
    paid SDK fallback."""
    return _matching_days(conn, message_ids, 'pi-agent')


# ── Meridian twin sessions (activity metrics) ──
# Meridian writes a separate Claude Code session for the requests it serves
# for Pi. Cost is deduped per row above; activity (sessions, prompts, active
# time, tools, agents) is per session, so the whole SDK copy is excluded. A
# Claude Code session is a twin exactly when one of its assistant rows shares
# an Anthropic msg_* ID with a Pi row of the same class: direct Claude Code
# use never carries a Pi response ID. Twin rows without a Pi counterpart
# (Meridian's own calls) still bill through the row predicate.
#
# Many SDK copies are caught before their reply is written and then deleted,
# leaving only the replayed prompt, attachments and tool results: no ID to
# match. Copies of failed Pi turns hold only a '<synthetic>' row, which Claude
# Code writes itself after an API error. A Claude Code session with no real
# model reply did no model work, so it is not activity either; a live session
# counts from its first reply.

_REPLY = "{a}.type='assistant' AND COALESCE({a}.model, '') != '<synthetic>'"

def counts_as_activity(alias: str = 'events') -> str:
    """Activity filter for events or tool_uses rows. Aliases are internal."""
    return (f"{not_pi_copy(alias)} AND NOT ({alias}.source_client='claude-code' AND {alias}.session_id IS NOT NULL "
            "AND (EXISTS (SELECT 1 FROM pi_mirror_sessions AS mirror "
            f"WHERE mirror.session_id={alias}.session_id) "
            "OR NOT EXISTS (SELECT 1 FROM events AS reply "
            f"WHERE reply.session_id={alias}.session_id AND {_REPLY.format(a='reply')})))")


# Predicate text matches idx_events_billable_ts (type='assistant' AND
# request_id IS NOT NULL) so the ts_epoch bound is an index range scan. A bound
# on SDK rows has no such path: idx_events_claude_message covers every Claude
# Code assistant row. INDEXED BY pins it: left alone the planner prefers the
# partial Pi message index, which walks every Pi row ever stored.
RECENT_PI_MESSAGES_SQL = (
    "SELECT DISTINCT message_id FROM events INDEXED BY idx_events_billable_ts "
    "WHERE type='assistant' "
    "AND request_id IS NOT NULL AND ts_epoch >= ? AND source_client='pi-agent' "
    "AND LOWER(provider)='anthropic' AND message_id IS NOT NULL")


def recent_pi_message_ids(conn: sqlite3.Connection, since_epoch: float) -> set[str]:
    """Anthropic response IDs of Pi rows since ``since_epoch`` (hourly repair)."""
    return {row[0] for row in conn.execute(RECENT_PI_MESSAGES_SQL, (since_epoch,))}


def find_mirror_sessions(conn: sqlite3.Connection,
                         message_ids: set[str] | None = None, *,
                         since_epoch: float | None = None) -> set[str]:
    """Twin sessions not yet recorded in pi_mirror_sessions. Read-only.

    ``message_ids=None`` scans all history (migration). Otherwise only SDK
    rows carrying those IDs are checked, which covers both arrival orders:
    an SDK batch passes its own IDs, a Pi batch passes the IDs it reported.
    ``since_epoch`` bounds the scan to recent SDK rows (hourly repair).
    """
    query = (
        "SELECT DISTINCT sdk.session_id FROM events AS sdk "
        "WHERE sdk.source_client='claude-code' AND sdk.type='assistant' "
        "AND sdk.session_id IS NOT NULL AND sdk.message_id GLOB 'msg_?*' "
        "AND NOT EXISTS (SELECT 1 FROM pi_mirror_sessions AS known "
        "WHERE known.session_id=sdk.session_id) "
        "AND EXISTS (SELECT 1 FROM events AS twin_pi WHERE "
        "twin_pi.source_client='pi-agent' AND LOWER(twin_pi.provider)='anthropic' "
        f"AND twin_pi.type='assistant' AND {_same_response('twin_pi', 'sdk')})")
    since = [] if since_epoch is None else [since_epoch]
    if since:
        query += " AND sdk.ts_epoch >= ?"
    if message_ids is None:
        batches = [None]
    else:
        ids = sorted(m for m in message_ids if m)
        batches = [ids[i:i + _MESSAGE_ID_BATCH]
                   for i in range(0, len(ids), _MESSAGE_ID_BATCH)]
    found: set[str] = set()
    for batch in batches:
        sql = query if batch is None else (
            query + f" AND sdk.message_id IN ({','.join('?' for _ in batch)})")
        found.update(row[0] for row in conn.execute(sql, since + (batch or [])))
    return found


def _insert_mirror_sessions(conn: sqlite3.Connection, session_ids: set[str]) -> None:
    conn.executemany("INSERT OR IGNORE INTO pi_mirror_sessions(session_id) VALUES (?)",
                     [(sid,) for sid in sorted(session_ids)])


def mark_mirror_sessions(conn: sqlite3.Connection) -> set[str]:
    """Record every twin session in history (migration; caller owns the txn)."""
    found = find_mirror_sessions(conn)
    _insert_mirror_sessions(conn, found)
    return found


def record_mirror_sessions(message_ids: set[str] | None = None, *,
                           since_epoch: float | None = None) -> set[str]:
    """Mark new twin sessions in their own transaction; return days to re-roll.

    Ingest calls this for every batch, an all-duplicate retry included, and
    the hourly sweep calls it for the last 7 days, which repairs a mark that
    failed after its events committed. Never raises: ingest is durable and
    the sweep timer must stay alive.
    """
    from .db import get_conn, write_txn
    if message_ids is not None and not message_ids:
        return set()
    try:
        # Scan outside the write lock; a race only repeats a harmless insert.
        found = find_mirror_sessions(get_conn(), message_ids, since_epoch=since_epoch)
        if not found:
            return set()
        with write_txn() as conn:
            _insert_mirror_sessions(conn, found)
        return session_days(get_conn(), found)
    except Exception:
        logger.exception("Meridian twin-session mark failed; the hourly sweep retries")
        return set()


def sessions_without_reply(conn: sqlite3.Connection, session_ids: set[str]) -> set[str]:
    """Sessions with no model reply yet: their first reply makes every day
    of the session count as activity (counts_as_activity)."""
    ids = sorted(session_ids)
    have: set[str] = set()
    for offset in range(0, len(ids), _MESSAGE_ID_BATCH):
        batch = ids[offset:offset + _MESSAGE_ID_BATCH]
        have.update(row[0] for row in conn.execute(
            f"SELECT DISTINCT session_id FROM events AS e WHERE {_REPLY.format(a='e')} "
            f"AND session_id IN ({','.join('?' for _ in batch)})", batch))
    return set(ids) - have


def session_days(conn: sqlite3.Connection, session_ids: set[str]) -> set[str]:
    """Every day a session touched: a new twin changes all of its rollups."""
    ids = sorted(session_ids)
    days: set[str] = set()
    for offset in range(0, len(ids), _MESSAGE_ID_BATCH):
        batch = ids[offset:offset + _MESSAGE_ID_BATCH]
        days.update(row[0] for row in conn.execute(
            f"SELECT DISTINCT day FROM events WHERE session_id IN "
            f"({','.join('?' for _ in batch)})", batch))
    return days
