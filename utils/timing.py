import time
from contextlib import contextmanager


@contextmanager
def timed_section(report_fn=None):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if report_fn:
        report_fn(elapsed)
