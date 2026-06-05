# timer.py
import time
from config_new import time_limit

_START_TIME = None
_TIME_LIMIT = time_limit

def start(time_limit_sec):
    """Initialize global wall-clock timer ONCE."""
    global _START_TIME, _TIME_LIMIT
    _START_TIME = time.perf_counter()
    _TIME_LIMIT = time_limit_sec


def elapsed():
    return time.perf_counter() - _START_TIME


def remaining():
    return max(0.0, _TIME_LIMIT - elapsed())


def expired(buffer=0.5):
    """Return True if time is (almost) exhausted."""
    return remaining() <= buffer