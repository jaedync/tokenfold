"""Read-side ownership boundary for append-only Claude quota observations.

Legacy server/client polls were one feed. They are not comparable to Meridian's
account/profile observations, even when bucket names and reset anchors match.
Never delete the old evidence or merely clamp a resulting negative slope.
"""
from .claude_usage import MANAGED_SOURCE, managed_source_owns_usage


def active_history_source(conn):
    """Resolve once per read operation, not independently for each SQL query."""
    return MANAGED_SOURCE if managed_source_owns_usage(conn) else None


def history_source_filter(source):
    """Static SQL plus bound parameters; None selects the legacy writer group.

    Filtering both ways also prevents a legacy observation's inference from
    seeing newly appended managed rows during the one-way ownership transfer.
    """
    if source == MANAGED_SOURCE:
        return "source=?", (MANAGED_SOURCE,)
    return "(source IS NULL OR source<>?)", (MANAGED_SOURCE,)
