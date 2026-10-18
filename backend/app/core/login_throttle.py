"""
In-process login throttling for the single-worker prototype.

Not shared across multiple workers — staging must run one API worker for
admin auth until a shared store is introduced.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class LoginThrottle:
    max_failures: int = 5
    window_seconds: int = 300
    _failures: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _prune(self, key: str, now: float) -> None:
        cutoff = now - float(self.window_seconds)
        self._failures[key] = [t for t in self._failures[key] if t >= cutoff]

    def is_blocked(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            self._prune(key, now)
            return len(self._failures[key]) >= self.max_failures

    def record_failure(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._prune(key, now)
            self._failures[key].append(now)

    def clear(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)


# Process-global throttle used by the admin login route.
default_login_throttle = LoginThrottle()
