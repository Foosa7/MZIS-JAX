"""Process-wide configuration and the objects the API and worker share.

The API process and the Celery worker are separate programs that must agree
on the chip, its calibration and its lease, so all of that is resolved from
environment variables in one place.

Only the worker opens the device. The API answers questions and hands out
leases; it never drives current itself, which keeps a single writer per chip
even when several API workers are running.
"""

import logging
import os
from functools import lru_cache

log = logging.getLogger("mzix.runtime")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class Settings:
    def __init__(self, env=None):
        env = env or os.environ

        self.redis_url = env.get("MZIX_REDIS_URL", "redis://localhost:6379/0")
        self.grid_size = env.get("MZIX_GRID_SIZE", "8x8")
        self.backend = env.get("MZIX_BACKEND", "simulator").lower()
        self.serial_port = env.get("MZIX_SERIAL_PORT") or None
        self.max_current_mA = float(env.get("MZIX_MAX_CURRENT_MA", "6.0"))
        self.n_channels = int(env.get("MZIX_N_CHANNELS", "64"))

        self.calibration_path = env.get(
            "MZIX_CALIBRATION",
            os.path.join(REPO_ROOT, "node-isolation", "8-mode-autocal-20260209.json"),
        )
        self.users_path = env.get(
            "MZIX_USERS_FILE", os.path.join(REPO_ROOT, "config", "users.json")
        )
        self.audit_path = env.get(
            "MZIX_AUDIT_LOG", os.path.join(REPO_ROOT, "var", "audit.jsonl")
        )
        self.audit_fsync = env.get("MZIX_AUDIT_FSYNC", "0") == "1"
        self.max_lease_seconds = int(env.get("MZIX_MAX_LEASE_SECONDS", "1800"))

        # Bind address. Defaulting to loopback rather than 0.0.0.0 matters:
        # the auth model assumes nothing but the tailnet can reach the port.
        self.bind_host = env.get("MZIX_BIND_HOST", "127.0.0.1")
        self.bind_port = int(env.get("MZIX_BIND_PORT", "8000"))

    @property
    def n_modes(self):
        return int(str(self.grid_size).split("x")[0])

    def describe(self):
        return {
            "grid_size": self.grid_size,
            "backend": self.backend,
            "max_current_mA": self.max_current_mA,
            "calibration": os.path.basename(self.calibration_path),
            "bind": f"{self.bind_host}:{self.bind_port}",
        }


@lru_cache(maxsize=1)
def get_settings():
    return Settings()


@lru_cache(maxsize=1)
def get_calibration():
    from utils.qontrol.calibration import HeaterCalibration

    settings = get_settings()
    calibration = HeaterCalibration.from_file(
        settings.calibration_path, max_current_mA=settings.max_current_mA
    )
    log.info("loaded calibration for %d heater(s)", len(calibration.calibrated_heaters()))
    return calibration


@lru_cache(maxsize=1)
def get_phase_controller():
    from utils.qontrol.phase_control import PhaseController

    settings = get_settings()
    return PhaseController(get_calibration(), grid_size=settings.grid_size)


@lru_cache(maxsize=1)
def get_audit_log():
    from src.server.audit import AuditLog

    settings = get_settings()
    return AuditLog(settings.audit_path, fsync=settings.audit_fsync)


@lru_cache(maxsize=1)
def get_redis():
    import redis

    return redis.Redis.from_url(get_settings().redis_url)


@lru_cache(maxsize=1)
def get_chip_lease():
    from src.server.lock import ChipLease

    settings = get_settings()
    return ChipLease(get_redis(), max_ttl_seconds=settings.max_lease_seconds)


@lru_cache(maxsize=1)
def get_quota_tracker():
    from src.server.lock import QuotaTracker

    return QuotaTracker(get_redis())


@lru_cache(maxsize=1)
def get_driver():
    """Opens the current driver. Worker-side only."""
    from utils.qontrol.device import RealQontrol, SafeDriver, SimulatedQontrol
    from utils.qontrol.mapping_utils import create_label_mapping

    settings = get_settings()

    if settings.backend in ("simulator", "sim", "simulated"):
        backend = SimulatedQontrol(
            n_channels=settings.n_channels,
            global_current_limit_mA=settings.max_current_mA,
            calibration=get_calibration(),
            label_map=create_label_mapping(settings.grid_size),
            n_modes=settings.n_modes,
        )
        log.warning("using the SIMULATED chip backend; no hardware will be driven")
    elif settings.backend in ("qontrol", "real", "hardware"):
        backend = RealQontrol(
            serial_port=settings.serial_port,
            global_current_limit_mA=settings.max_current_mA,
        )
        log.info("using the real Qontrol backend on %s", settings.serial_port or "auto")
    else:
        raise ValueError(f"unknown MZIX_BACKEND {settings.backend!r}")

    driver = SafeDriver(backend, audit=get_audit_log(), max_current_mA=settings.max_current_mA)
    driver.connect()
    return driver
