import logging
import time

import pytest

from src import rate_limit
from src.guard import GuardError
from src.rate_limit import RateLimitError, check_rate_limit

# Ten requests one second apart, from t=0 to t=9: the per-minute limit is used up.
FULL_MINUTE = [float(t) for t in range(10)]


def test_ten_requests_in_a_minute_pass():
    """Each of the first 10 requests in one minute is allowed."""
    for sent in range(10):
        check_rate_limit(FULL_MINUTE[:sent], now=9.5)


def test_eleventh_request_is_blocked_with_the_wait_time():
    """The 11th request in a minute is blocked until the oldest one is 60 s old."""
    # The oldest request (t=0) leaves the window at t=60, 50.5 s from now: rounded up to 51.
    with pytest.raises(RateLimitError, match="Please wait 51 seconds before sending another"):
        check_rate_limit(FULL_MINUTE, now=9.5)


def test_wait_under_a_second_is_one_second():
    """A wait of a fraction of a second is shown as "1 second", not "0 seconds"."""
    with pytest.raises(RateLimitError, match="Please wait 1 second before"):
        check_rate_limit(FULL_MINUTE, now=59.9)


def test_window_resets_after_60_seconds():
    """Once the oldest request is 60 s old, one more request is allowed."""
    check_rate_limit(FULL_MINUTE, now=60.0)


def test_window_is_rolling():
    """Only requests from the last 60 s count, so each expiring one frees exactly one slot."""
    # At t=61 the requests at t=0 and t=1 have left the window: 8 left, so 2 more may go.
    times = [*FULL_MINUTE, 61.0]
    check_rate_limit(times, now=61.0)
    with pytest.raises(RateLimitError, match="Please wait 1 second before"):
        check_rate_limit([*times, 61.0], now=61.0)


def test_session_cap_holds_even_with_an_empty_window():
    """After 50 requests, spread out so the minute limit never fires, the next is blocked."""
    times = [i * 120.0 for i in range(50)]
    check_rate_limit(times[:49], now=10_000.0)
    with pytest.raises(RateLimitError) as caught:
        check_rate_limit(times, now=10_000.0)
    assert str(caught.value) == "You have used all 50 messages for this session."


def test_session_cap_is_checked_before_the_minute_limit():
    """With both limits reached, the session message is shown: waiting would not help."""
    times = [0.0] * 50
    with pytest.raises(RateLimitError, match="all 50 messages"):
        check_rate_limit(times, now=1.0)


def test_limits_can_be_changed():
    """The limits and the window are parameters, defaulting to the values in config.py."""
    check_rate_limit([0.0, 1.0], now=2.0, per_minute=3)
    with pytest.raises(RateLimitError, match="sent 2 messages.*wait 8 seconds"):
        check_rate_limit([0.0, 1.0], now=2.0, per_minute=2, window=10)
    with pytest.raises(RateLimitError, match="all 2 messages"):
        check_rate_limit([0.0, 1.0], now=500.0, per_session=2)


def test_wait_counts_every_request_over_the_limit():
    """With more requests in the window than the limit, the wait lasts until enough have left."""
    # Limit 2, window 10 s, requests at t=0, 1, 2: a slot opens only when t=1 leaves, at t=11.
    with pytest.raises(RateLimitError, match="wait 8 seconds"):
        check_rate_limit([0.0, 1.0, 2.0], now=3.0, per_minute=2, window=10)
    with pytest.raises(RateLimitError, match="wait 1 second "):
        check_rate_limit([0.0, 1.0, 2.0], now=10.0, per_minute=2, window=10)
    check_rate_limit([0.0, 1.0, 2.0], now=11.0, per_minute=2, window=10)


def test_clock_is_monotonic():
    """The app's clock is `time.monotonic`, so changing the system time cannot skip the limit."""
    assert rate_limit.clock is time.monotonic


def test_rate_limit_error_is_a_guard_error():
    """The app's existing GuardError handling also catches a rate-limit block."""
    assert issubclass(RateLimitError, GuardError)


@pytest.mark.parametrize(
    ("times", "reason"),
    [(FULL_MINUTE, "per_minute"), ([0.0] * 50, "per_session")],
    ids=["per-minute", "per-session"],
)
def test_block_is_logged(caplog, times, reason):
    """Each block logs one WARNING on `src.rate_limit` naming the limit that was hit."""
    with caplog.at_level(logging.WARNING, logger="src.rate_limit"):
        with pytest.raises(RateLimitError):
            check_rate_limit(times, now=9.5)
    records = [r for r in caplog.records if r.name == "src.rate_limit"]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert reason in records[0].getMessage()


def test_allowed_request_logs_nothing(caplog):
    """A request under both limits is not logged."""
    with caplog.at_level(logging.DEBUG, logger="src.rate_limit"):
        check_rate_limit(FULL_MINUTE[:9], now=9.5)
    assert not [r for r in caplog.records if r.name == "src.rate_limit"]
