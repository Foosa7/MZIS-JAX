"""Exclusive access to the chip.

One chip, one writer. Without this, two users' jobs interleave current writes
and both get physics that reflects neither request -- and neither user can
tell, because each sees only their own telemetry.

The lease is a Redis key with a TTL and a holder token. A crashed client
therefore frees the chip on its own rather than wedging the bench, while a
live one keeps the lease by renewing.

Compare-and-act runs inside a WATCH/MULTI transaction, so a renew or release
cannot land on a lease that expired and was retaken between the read and the
write. (WATCH rather than a Lua script: the semantics are identical here and
it works on any redis-py-compatible client.)
"""

import json
import secrets
import time

LEASE_KEY = "mzix:chip:lease"


class LeaseError(RuntimeError):
    """Raised when an operation is attempted without a valid lease."""


class ChipLease:
    def __init__(self, redis_client, key=LEASE_KEY, max_ttl_seconds=1800):
        self.redis = redis_client
        self.key = key
        self.max_ttl_seconds = max_ttl_seconds

    # ── inspection ───────────────────────────────────────────────────────

    def holder(self):
        """The current lease, or None if the chip is free."""
        raw = self.redis.get(self.key)
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode()
        return json.loads(raw)

    def held_by(self, login):
        lease = self.holder()
        return bool(lease and lease.get("user") == login)

    # ── mutation ─────────────────────────────────────────────────────────

    def acquire(self, login, ttl_seconds=None, note=None):
        """Takes the chip. Returns the lease, or None if someone else holds it.

        Re-acquiring while already the holder extends rather than fails, so a
        client that lost its token is not locked out of its own lease.
        """
        ttl = min(int(ttl_seconds or self.max_ttl_seconds), self.max_ttl_seconds)
        if ttl <= 0:
            raise LeaseError("this role may not hold the chip")

        now = time.time()
        lease = {
            "user": login,
            "token": secrets.token_urlsafe(24),
            "acquired_at": now,
            "expires_at": now + ttl,
            "note": note,
        }

        if self.redis.set(self.key, json.dumps(lease), nx=True, px=ttl * 1000):
            return lease

        existing = self.holder()
        if existing and existing.get("user") == login:
            existing["expires_at"] = now + ttl
            self.redis.set(self.key, json.dumps(existing), px=ttl * 1000)
            return existing
        return None

    def _compare_and_act(self, token, action):
        """Runs `action(pipe, lease)` only while the lease still has `token`."""

        def attempt(pipe):
            raw = pipe.get(self.key)
            if not raw:
                return False
            if isinstance(raw, bytes):
                raw = raw.decode()
            lease = json.loads(raw)
            if lease.get("token") != token:
                return False
            pipe.multi()
            action(pipe, lease)
            return True

        return bool(self.redis.transaction(attempt, self.key, value_from_callable=True))

    def renew(self, token, ttl_seconds=None):
        ttl = min(int(ttl_seconds or self.max_ttl_seconds), self.max_ttl_seconds)

        def extend(pipe, lease):
            lease["expires_at"] = time.time() + ttl
            pipe.set(self.key, json.dumps(lease), px=ttl * 1000)

        return self._compare_and_act(token, extend)

    def release(self, token):
        return self._compare_and_act(token, lambda pipe, lease: pipe.delete(self.key))

    def force_release(self):
        """Admin override for a lease whose holder has gone away."""
        previous = self.holder()
        self.redis.delete(self.key)
        return previous

    # ── enforcement ──────────────────────────────────────────────────────

    def require(self, login, token=None):
        """Raises unless `login` currently holds the chip.

        Called by the worker immediately before touching hardware, so a lease
        that lapsed mid-job stops the next write rather than the next job.
        """
        lease = self.holder()
        if lease is None:
            raise LeaseError("no chip lease is held; acquire one before driving hardware")
        if lease.get("user") != login:
            raise LeaseError(f"chip is leased to {lease.get('user')}, not {login}")
        if token is not None and lease.get("token") != token:
            raise LeaseError("lease token is stale; the lease was retaken")
        return lease


# ──────────────────────────────────────────────────────────────────────────────
# Quotas
# ──────────────────────────────────────────────────────────────────────────────

class QuotaTracker:
    """Per-user job counters, expiring daily."""

    def __init__(self, redis_client, prefix="mzix:quota"):
        self.redis = redis_client
        self.prefix = prefix

    def _day_key(self, login):
        day = time.strftime("%Y%m%d", time.gmtime())
        return f"{self.prefix}:{login}:{day}"

    def _running_key(self, login):
        return f"{self.prefix}:running:{login}"

    def jobs_today(self, login):
        return int(self.redis.get(self._day_key(login)) or 0)

    def running(self, login):
        return int(self.redis.get(self._running_key(login)) or 0)

    def check(self, principal):
        """Raises LeaseError when the caller is out of budget."""
        quotas = principal.quotas
        max_daily = quotas.get("max_jobs_per_day", 0)
        max_running = quotas.get("max_concurrent_jobs", 0)

        if max_running <= 0:
            raise LeaseError(f"role {principal.role!r} may not submit jobs")
        if self.running(principal.login) >= max_running:
            raise LeaseError(
                f"{principal.login} already has {max_running} job(s) in flight"
            )
        if max_daily and self.jobs_today(principal.login) >= max_daily:
            raise LeaseError(
                f"{principal.login} reached the daily limit of {max_daily} jobs"
            )

    def record_submit(self, login):
        day_key = self._day_key(login)
        self.redis.incr(day_key)
        self.redis.expire(day_key, 2 * 24 * 3600)
        self.redis.incr(self._running_key(login))

    def record_finish(self, login):
        # Never let the running counter drift below zero on a double-finish.
        if self.redis.decr(self._running_key(login)) < 0:
            self.redis.set(self._running_key(login), 0)
