import time
from contextlib import contextmanager


@contextmanager
def timed_section(name=None, report_fn=None):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if report_fn:
            report_fn(elapsed)
        else:
            if name:
                print(f"[{name}] Elapsed: {elapsed:.4f} seconds")
            else:
                print(f"Elapsed: {elapsed:.4f} seconds")
