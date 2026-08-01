"""MZIX control API.

Every route is authenticated by Tailscale identity and gated on a role. Two
holes in the previous version are closed here:

* `/api/v1/jobs` took `user_id` from the request body, so anyone could submit
  work as anyone else. Identity now comes from the tailnet and the body field
  is ignored.
* `/ws/telemetry/{user_id}` let any client subscribe to any user's stream by
  guessing their id. The socket now streams only the authenticated caller's
  own channel; the path parameter is gone.

The API never drives current. It hands out leases and queues work; the worker
is the only writer, which is what keeps a single writer per chip.
"""

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status

from src.server.auth import (
    PERMISSIONS,
    Principal,
    UserDirectory,
    current_user,
    identify,
    require,
)
from src.server.lock import LeaseError
from src.server.models import (
    ChipStatus,
    JobPayload,
    JobRequest,
    LeaseRequest,
    LeaseToken,
    WhoAmI,
)
from src.server.runtime import (
    get_audit_log,
    get_calibration,
    get_chip_lease,
    get_quota_tracker,
    get_settings,
)

log = logging.getLogger("mzix.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.users = UserDirectory(settings.users_path)
    log.info("MZIX control API up: %s", settings.describe())
    if settings.bind_host in ("0.0.0.0", "::"):
        log.error(
            "MZIX_BIND_HOST is %s. The auth model assumes only the tailnet can "
            "reach this port; bind to the Tailscale address instead.",
            settings.bind_host,
        )
    yield


app = FastAPI(title="MZIX Control API", version="2.0", lifespan=lifespan)


# ──────────────────────────────────────────────────────────────────────────────
# Identity
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/me", response_model=WhoAmI)
async def whoami(user: Principal = Depends(current_user)):
    quotas = get_quota_tracker()
    return WhoAmI(
        login=user.login,
        display_name=user.display_name,
        role=user.role,
        address=user.address,
        permissions=sorted(PERMISSIONS.get(user.role, set())),
        quotas=user.quotas,
        jobs_today=quotas.jobs_today(user.login),
        jobs_running=quotas.running(user.login),
    )


@app.get("/api/v1/users")
async def list_users(user: Principal = Depends(require("users:read"))):
    return {"users": app.state.users.all_users()}


# ──────────────────────────────────────────────────────────────────────────────
# Chip status and lease
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/chip/status", response_model=ChipStatus)
async def chip_status(user: Principal = Depends(require("chip:read"))):
    settings = get_settings()
    lease = get_chip_lease().holder()
    if lease:
        # The token is the holder's capability; never hand it to onlookers.
        lease = {k: v for k, v in lease.items() if k != "token"}

    calibration = get_calibration()
    calibrated = set(calibration.calibrated_heaters())
    uncalibrated = sorted(set(calibration.phase_calibration) - calibrated)

    return ChipStatus(
        backend=settings.backend,
        grid_size=settings.grid_size,
        n_channels=settings.n_channels,
        max_current_mA=settings.max_current_mA,
        lease=lease,
        calibrated_heaters=len(calibrated),
        uncalibrated_heaters=uncalibrated,
    )


@app.post("/api/v1/chip/lease")
async def acquire_lease(request: LeaseRequest,
                        user: Principal = Depends(require("chip:lease"))):
    ttl_cap = user.quotas.get("max_lease_seconds", 0)
    if ttl_cap <= 0:
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            f"role {user.role!r} may not hold the chip")
    ttl = min(request.ttl_seconds or ttl_cap, ttl_cap)

    try:
        lease = get_chip_lease().acquire(user.login, ttl_seconds=ttl, note=request.note)
    except LeaseError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc))

    if lease is None:
        holder = get_chip_lease().holder() or {}
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"chip is in use by {holder.get('user', 'someone else')} "
            f"until {holder.get('expires_at')}",
        )

    get_audit_log().record("lease_acquired", user=user.login, ttl_seconds=ttl,
                           note=request.note)
    return lease


@app.post("/api/v1/chip/lease/renew")
async def renew_lease(token: LeaseToken,
                      user: Principal = Depends(require("chip:lease"))):
    ttl = user.quotas.get("max_lease_seconds", 0)
    if not get_chip_lease().renew(token.token, ttl_seconds=ttl):
        raise HTTPException(status.HTTP_409_CONFLICT,
                            "lease is not yours or has already expired")
    return get_chip_lease().holder()


@app.delete("/api/v1/chip/lease")
async def release_lease(token: LeaseToken,
                        user: Principal = Depends(require("chip:lease"))):
    if not get_chip_lease().release(token.token):
        raise HTTPException(status.HTTP_409_CONFLICT,
                            "lease is not yours or has already expired")
    get_audit_log().record("lease_released", user=user.login)
    return {"status": "released"}


@app.delete("/api/v1/chip/lease/force")
async def force_release_lease(user: Principal = Depends(require("chip:force_release"))):
    previous = get_chip_lease().force_release()
    get_audit_log().record("lease_force_released", user=user.login,
                           previous_holder=(previous or {}).get("user"))
    return {"status": "released", "previous": previous}


@app.post("/api/v1/chip/emergency-stop")
async def emergency_stop(user: Principal = Depends(require("chip:stop"))):
    """Zeroes every channel. Available to any operator, lease or no lease."""
    from src.server.worker import emergency_stop as stop_task

    task = stop_task.apply_async(args=[user.login], priority=0)
    get_audit_log().record("emergency_stop_requested", user=user.login)
    return {"status": "requested", "task_id": task.id}


# ──────────────────────────────────────────────────────────────────────────────
# Jobs
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/jobs")
async def submit_job(request: JobRequest,
                     user: Principal = Depends(require("job:submit"))):
    lease = get_chip_lease()
    held = lease.holder()
    if not held or held.get("user") != user.login:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "acquire the chip lease before submitting work"
            + (f"; currently held by {held['user']}" if held else ""),
        )

    quotas = get_quota_tracker()
    try:
        quotas.check(user)
    except LeaseError as exc:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(exc))

    from src.server.worker import process_batch_job

    payload = JobPayload(
        job_id=request.job_id or uuid.uuid4().hex,
        user_id=user.login,
        lease_token=held.get("token"),
        priority=request.priority,
        unitaries=request.unitaries,
    )

    quotas.record_submit(user.login)
    try:
        task = process_batch_job.apply_async(args=[payload.model_dump()],
                                             priority=payload.priority)
    except Exception:
        # Never leave the in-flight counter charged for a job that never queued.
        quotas.record_finish(user.login)
        raise

    get_audit_log().record("job_queued", user=user.login, job_id=payload.job_id,
                           targets=len(payload.unitaries), task_id=task.id)
    return {"status": "queued", "task_id": task.id, "job_id": payload.job_id}


@app.get("/api/v1/jobs/{task_id}")
async def job_status(task_id: str, user: Principal = Depends(require("chip:read"))):
    from src.server.worker import app as celery_app

    result = celery_app.AsyncResult(task_id)
    payload = {"task_id": task_id, "state": result.state}
    if result.ready():
        try:
            payload["result"] = result.get(timeout=1)
        except Exception as exc:
            payload["error"] = str(exc)
    return payload


@app.get("/api/v1/audit")
async def read_audit(limit: int = 100,
                     user: Principal = Depends(require("audit:read"))):
    return {"entries": get_audit_log().tail(limit=min(limit, 1000))}


# ──────────────────────────────────────────────────────────────────────────────
# Telemetry
# ──────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/telemetry")
async def telemetry(websocket: WebSocket):
    """Streams the authenticated caller's own telemetry.

    Authentication happens before the socket is accepted, and the channel is
    derived from the verified identity rather than from the URL.
    """
    client = websocket.client
    try:
        user = identify(client.host if client else None,
                        client.port if client else 0,
                        websocket.app.state.users)
    except HTTPException as exc:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION,
                              reason=str(exc.detail)[:120])
        return

    if not user.can("chip:read"):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="not authorised")
        return

    await websocket.accept()
    settings = get_settings()
    redis_conn = aioredis.from_url(settings.redis_url)
    pubsub = redis_conn.pubsub()
    channel = f"telemetry:{user.login}"
    await pubsub.subscribe(channel)

    async def pump():
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message:
                data = message["data"]
                if isinstance(data, bytes):
                    data = data.decode()
                await websocket.send_json(json.loads(data))
            await asyncio.sleep(0.01)

    pump_task = asyncio.create_task(pump())
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        pump_task.cancel()
        await pubsub.unsubscribe(channel)
        await pubsub.close()
        await redis_conn.close()
