"""Bounded FIFO admission around a model whose hooks are process-global."""
from collections import deque
from contextlib import contextmanager
import threading
import time


class BusyError(Exception):
    pass


class TaskGate:
    def __init__(self, max_queue=4, wait_timeout=30):
        self.max_queue = max_queue
        self.wait_timeout = wait_timeout
        self._condition = threading.Condition()
        self._waiting = deque()
        self._active = False

    @property
    def waiting(self):
        with self._condition:
            return len(self._waiting)

    @contextmanager
    def enter(self):
        ticket = object()
        deadline = time.monotonic() + self.wait_timeout
        with self._condition:
            if (self._active or self._waiting) and len(self._waiting) >= self.max_queue:
                raise BusyError("model queue full; retry later")
            self._waiting.append(ticket)
            while self._active or self._waiting[0] is not ticket:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._waiting.remove(ticket)
                    self._condition.notify_all()
                    raise BusyError("model queue wait timed out; retry later")
                self._condition.wait(remaining)
            self._waiting.popleft()
            self._active = True
        try:
            yield
        finally:
            with self._condition:
                self._active = False
                self._condition.notify_all()
