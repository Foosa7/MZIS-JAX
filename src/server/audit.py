"""Append-only record of everything that reached the hardware.

Shared bench equipment needs an answer to "who set that current, and when".
Records are written before the corresponding device write, so the log can
contain an attempt that failed but never omit one that succeeded.

One JSON object per line: greppable, appendable, and safe to tail while the
server writes to it.
"""

import json
import os
import threading
import time


class AuditLog:
    def __init__(self, path, fsync=False):
        self.path = path
        self.fsync = fsync
        self._lock = threading.Lock()

        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)

    def record(self, action, user=None, job_id=None, **fields):
        entry = {
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "action": action,
            "user": user,
            "job_id": job_id,
        }
        entry.update(fields)

        line = json.dumps(entry, default=str, sort_keys=True)
        with self._lock:
            with open(self.path, "a") as f:
                f.write(line + "\n")
                f.flush()
                if self.fsync:
                    os.fsync(f.fileno())
        return entry

    def tail(self, limit=100, user=None):
        """Most recent entries, newest last. For the admin audit endpoint."""
        try:
            with open(self.path) as f:
                lines = f.readlines()
        except FileNotFoundError:
            return []

        entries = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if user is None or entry.get("user") == user:
                entries.append(entry)
        return entries[-limit:]


class NullAuditLog:
    """Used in tests and dry runs; keeps the call sites unconditional."""

    def record(self, action, user=None, job_id=None, **fields):
        return None

    def tail(self, limit=100, user=None):
        return []
