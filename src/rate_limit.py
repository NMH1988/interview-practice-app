"""Rate limiting: caps how many requests one browser session may send to the LLM."""

import logging
import math
import time
from collections.abc import Sequence

from src.config import RATE_LIMIT_PER_MINUTE, RATE_LIMIT_PER_SESSION, RATE_LIMIT_WINDOW_SECONDS
from src.guard import GuardError

logger = logging.getLogger(__name__)

# Monotonic, so a change to the system clock cannot open or stretch the window. The app calls it
# through the module (`rate_limit.clock()`), so tests can replace it with a fake clock.
clock = time.monotonic


class RateLimitError(GuardError):
    """Raised when a session has sent too many requests; `str(exc)` says how long to wait."""


def check_rate_limit(
    times: Sequence[float],
    now: float,
    *,
    per_minute: int = RATE_LIMIT_PER_MINUTE,
    per_session: int = RATE_LIMIT_PER_SESSION,
    window: float = RATE_LIMIT_WINDOW_SECONDS,
) -> None:
    """Raise `RateLimitError` if one more request now would go over a limit, given past `times`."""
    if len(times) >= per_session:
        logger.warning("Rate limited: per_session (%d requests)", len(times))
        raise RateLimitError(f"You have used all {per_session} messages for this session.")
    # Rolling window: a request stops counting exactly `window` seconds after it was sent.
    recent = sorted(t for t in times if now - t < window)
    if len(recent) >= per_minute:
        # One more may be sent once the window holds one less than the limit, i.e. once the
        # first len(recent) - per_minute + 1 requests have left it. Usually that is just the
        # oldest, but more if the window is over the limit (e.g. the limit was lowered).
        wait = max(1, math.ceil(recent[len(recent) - per_minute] + window - now))
        unit = "second" if wait == 1 else "seconds"
        logger.warning("Rate limited: per_minute (wait %d s)", wait)
        raise RateLimitError(
            f"You've sent {len(recent)} messages in the last minute. "
            f"Please wait {wait} {unit} before sending another."
        )
