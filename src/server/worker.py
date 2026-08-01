"""Celery worker: target unitary in, currents on the chip out.

What was here before was a mock -- `time.sleep(0.5)` and a synthetic falling
loss -- that never touched hardware and carried its own hardcoded thermal
constants instead of the per-heater calibration.

The real pipeline is:

    U_target -> Clements decomposition -> engine phases -> per-heater currents
             -> lease check -> device write -> measurement -> telemetry

Concurrency is pinned to one worker process per chip. That is not a
performance choice: two workers would interleave writes to the same DACs.
The lease is re-checked immediately before every write, so a lease that
lapses mid-job stops the next write rather than the next job.
"""

import json
import logging
import os
import time

import numpy as np
from celery import Celery

from src.server.lock import LeaseError
from src.server.runtime import (
    get_audit_log,
    get_chip_lease,
    get_driver,
    get_phase_controller,
    get_quota_tracker,
    get_redis,
    get_settings,
)

log = logging.getLogger("mzix.worker")

_settings = get_settings()

app = Celery(
    "mzix_worker",
    broker=os.environ.get("MZIX_BROKER_URL", _settings.redis_url),
    backend=os.environ.get("MZIX_RESULT_URL", "redis://localhost:6379/1"),
)

app.conf.broker_transport_options = {
    "priority_steps": list(range(10)),
    "queue_order_strategy": "priority",
}
# One chip, one writer.
app.conf.worker_concurrency = 1
app.conf.worker_prefetch_multiplier = 1
app.conf.task_acks_late = True


def publish(user_id, payload):
    """Pushes a telemetry frame to whichever socket that user has open."""
    try:
        get_redis().publish(f"telemetry:{user_id}", json.dumps(payload))
    except Exception as exc:
        log.warning("telemetry publish failed: %s", exc)


# ──────────────────────────────────────────────────────────────────────────────
# Target -> phases
# ──────────────────────────────────────────────────────────────────────────────

def target_to_grid_phases(matrix, n_modes):
    """Decomposes a target unitary into per-MZI phases in radians.

    Uses the common-mode-corrected mapping; the naive `2*theta_pnn` version
    silently misprograms anything that is not a permutation.
    """
    from decompose.pnn import clements_to_engine_phases, decompose_clements
    from src.engine import Engine

    engine = Engine(n_modes=n_modes)
    phis, thetas, alphas = decompose_clements(matrix.copy(), block="mzi")
    settings, output_phases = clements_to_engine_phases(phis, thetas, alphas, engine.layout)

    grid = {mzi_id: {"theta": float(theta), "phi": float(phi)}
            for mzi_id, (theta, phi) in settings.items()}
    return grid, output_phases, engine


def _as_matrix(target):
    real = np.asarray(target["matrix_real"], dtype=float)
    imag = np.asarray(target["matrix_imag"], dtype=float)
    if real.shape != imag.shape:
        raise ValueError("matrix_real and matrix_imag have different shapes")
    if real.ndim != 2 or real.shape[0] != real.shape[1]:
        raise ValueError(f"target must be square, got {real.shape}")

    matrix = real + 1j * imag
    deviation = np.abs(matrix.conj().T @ matrix - np.eye(matrix.shape[0])).max()
    if deviation > 1e-6:
        raise ValueError(f"target is not unitary (deviation {deviation:.2e})")
    return matrix


# ──────────────────────────────────────────────────────────────────────────────
# Task
# ──────────────────────────────────────────────────────────────────────────────

@app.task(bind=True)
def process_batch_job(self, payload_dict):
    job_id = payload_dict["job_id"]
    user_id = payload_dict["user_id"]
    targets = payload_dict["unitaries"]
    lease_token = payload_dict.get("lease_token")

    audit = get_audit_log()
    lease = get_chip_lease()
    quotas = get_quota_tracker()
    controller = get_phase_controller()
    settings = get_settings()

    audit.record("job_start", user=user_id, job_id=job_id, targets=len(targets),
                 backend=settings.backend)
    results = []

    try:
        driver = get_driver()

        for index, target in enumerate(targets):
            name = target.get("name", f"target_{index}")

            # Re-checked per target: a lease that lapsed must stop the next write.
            lease.require(user_id, token=lease_token)

            matrix = _as_matrix(target)
            if matrix.shape[0] != settings.n_modes:
                raise ValueError(
                    f"{name}: target is {matrix.shape[0]}x{matrix.shape[0]} but the "
                    f"chip is {settings.grid_size}"
                )

            grid, output_phases, _engine = target_to_grid_phases(matrix, settings.n_modes)
            grid_currents, applied, skipped = controller.apply_phases_to_hardware(
                driver, grid, user=user_id, job_id=job_id
            )

            measured, fidelity = None, None
            backend = driver.backend
            if hasattr(backend, "measured_output_powers"):
                measured = backend.measured_output_powers()
                if measured is not None:
                    expected = (np.abs(matrix) ** 2)[:, 0]
                    measured_arr = np.asarray(measured, dtype=float)
                    total = measured_arr.sum()
                    if total > 0:
                        measured_arr = measured_arr / total
                    # Classical (Bhattacharyya) fidelity between power profiles.
                    fidelity = float(np.sum(np.sqrt(expected * measured_arr)) ** 2)

            publish(user_id, {
                "job_id": job_id,
                "target": name,
                "index": index + 1,
                "total": len(targets),
                "heaters_written": len(applied),
                "heaters_skipped": len(skipped),
                "measured_powers": measured,
                "fidelity": fidelity,
                "residual_output_phases": np.asarray(output_phases).tolist(),
                "ts": time.time(),
            })

            audit.record("target_applied", user=user_id, job_id=job_id, target=name,
                         heaters=len(applied), skipped=len(skipped), fidelity=fidelity)

            results.append({
                "target": name,
                "fidelity": fidelity,
                "heaters_written": len(applied),
                "heaters_skipped": skipped,
                "currents": grid_currents,
            })

        audit.record("job_complete", user=user_id, job_id=job_id, targets=len(results))
        publish(user_id, {"job_id": job_id, "status": "completed", "ts": time.time()})
        return {"status": "completed", "job_id": job_id, "results": results}

    except LeaseError as exc:
        # Lost the chip. Someone else may already be driving it, so do not
        # touch the outputs on the way out.
        audit.record("job_aborted_lease", user=user_id, job_id=job_id, error=str(exc))
        publish(user_id, {"job_id": job_id, "status": "aborted", "error": str(exc)})
        raise

    except Exception as exc:
        audit.record("job_failed", user=user_id, job_id=job_id, error=str(exc))
        publish(user_id, {"job_id": job_id, "status": "failed", "error": str(exc)})
        # We still hold the lease here, so parking the heaters is ours to do
        # and is the safe state to leave the chip in.
        try:
            if lease.held_by(user_id):
                get_driver().zero_all(user=user_id, reason="zero_on_failure")
        except Exception as cleanup_exc:
            log.error("failed to park heaters after job error: %s", cleanup_exc)
        raise

    finally:
        quotas.record_finish(user_id)


@app.task
def emergency_stop(user_id="system"):
    """Drops every channel to zero, lease or no lease."""
    get_audit_log().record("emergency_stop", user=user_id)
    get_driver().zero_all(user=user_id, reason="emergency_stop")
    return {"status": "stopped"}
