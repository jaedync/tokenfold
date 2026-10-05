"""Canonical event filters shared by live and rolled-up usage accounting."""

import sqlite3
from .config import enterprise_predicate

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
    sdk = (
        f"{alias}.source_client='claude-code' AND {alias}.message_id GLOB 'msg_?*' "
        f"AND ({alias}.provider IS NULL OR LOWER({alias}.provider)='anthropic') "
        "AND EXISTS (SELECT 1 FROM events AS canonical_pi WHERE "
        f"{_billable('canonical_pi')} AND canonical_pi.source_client='pi-agent' "
        "AND LOWER(canonical_pi.provider)='anthropic' "
        f"AND {_positive_reported_cost('canonical_pi')} "
        f"AND {_same_response('canonical_pi', alias)})"
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
    return f'NOT ({sdk}) AND NOT ({pi})'


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
