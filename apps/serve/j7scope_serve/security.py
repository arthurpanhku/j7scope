"""Single-operator authentication and bounded, ephemeral research sessions."""
from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    session: str
    operator: bool = False


class Credentials:
    def __init__(self, api_key=None, *, ttl=86400, max_sessions=128):
        self.api_key = api_key
        self.ttl = ttl
        self.max_sessions = max_sessions
        self._tokens = {}
        self._lock = threading.Lock()

    def _prune(self):
        now = time.monotonic()
        self._tokens = {k: v for k, v in self._tokens.items() if v[1] > now}

    def resolve(self, token):
        if not self.api_key:
            return Principal("local", True)
        if token and hmac.compare_digest(token.encode(), self.api_key.encode()):
            return Principal("operator", True)
        if not token:
            return None
        key = hashlib.sha256(token.encode()).hexdigest()
        with self._lock:
            self._prune()
            item = self._tokens.get(key)
            return item[0] if item else None

    def issue(self, principal=None):
        with self._lock:
            self._prune()
            if len(self._tokens) >= self.max_sessions:
                raise ValueError("session limit reached; retry after sessions expire")
            principal = principal or Principal(secrets.token_hex(16))
            token = secrets.token_urlsafe(32)
            self._tokens[hashlib.sha256(token.encode()).hexdigest()] = (
                principal, time.monotonic() + self.ttl,
            )
            return token, principal

    def revoke(self, token):
        with self._lock:
            self._tokens.pop(hashlib.sha256(token.encode()).hexdigest(), None)
