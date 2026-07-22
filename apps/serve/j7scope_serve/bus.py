"""Thread-safe fan-out of J-space events to connected SSE viewers.

The server runs one chat request per thread (ThreadingHTTPServer); each viewer
holds another thread blocked on its own queue. The chat thread publishes token
events, the bus copies each event into every subscriber's queue. A small ring
buffer lets a viewer that connects mid-generation catch up on recent events.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from typing import Deque, Dict, List


class ReadoutBus:
    def __init__(self, *, backlog: int = 64, queue_max: int = 1024):
        self._lock = threading.Lock()
        self._subscribers: Dict[int, "queue.Queue[dict]"] = {}
        self._next_id = 0
        self._backlog: Deque[dict] = deque(maxlen=backlog)
        self._queue_max = queue_max

    def publish(self, event: dict) -> None:
        with self._lock:
            self._backlog.append(event)
            subs = list(self._subscribers.values())
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                # A stuck/slow viewer must never stall generation; drop for it.
                pass

    def subscribe(self, *, replay: bool = True) -> "tuple[int, queue.Queue[dict]]":
        q: "queue.Queue[dict]" = queue.Queue(maxsize=self._queue_max)
        with self._lock:
            sub_id = self._next_id
            self._next_id += 1
            self._subscribers[sub_id] = q
            if replay:
                for event in list(self._backlog):
                    try:
                        q.put_nowait(event)
                    except queue.Full:
                        break
        return sub_id, q

    def unsubscribe(self, sub_id: int) -> None:
        with self._lock:
            self._subscribers.pop(sub_id, None)

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)
