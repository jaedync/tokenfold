"""Shared inference policy; receipt success is never a quota observation."""
import math

QUOTA_STALE_AFTER_S = 3600
MAX_OBSERVATION_AGE_S = 24 * 3600
MAX_CLOCK_SKEW_S = 300


def fresh_observation(observed, now):
    return (not isinstance(observed, bool) and isinstance(observed, (int, float))
            and math.isfinite(observed) and observed > 0
            and 0 <= now - observed <= QUOTA_STALE_AFTER_S)


def quota_window_valid(observed, now, reset, start):
    """Only a current window containing a recent observation supports inference."""
    return (fresh_observation(observed, now)
            and all(isinstance(v, (int, float)) and math.isfinite(v)
                    for v in (reset, start))
            and reset > now and start < observed < reset)
