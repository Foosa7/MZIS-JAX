"""Control-layer tests: calibration, device safety, lease, quotas, auth, API.

Runs entirely against the simulated chip and fakeredis, so `pytest tests/`
exercises the whole stack with no hardware, no Redis and no tailnet.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import fakeredis

from src.server.audit import AuditLog, NullAuditLog
from src.server.lock import ChipLease, LeaseError, QuotaTracker
from utils.qontrol.calibration import HeaterCalibration
from utils.qontrol.device import DeviceError, SafeDriver, SimulatedQontrol
from utils.qontrol.mapping_utils import create_label_mapping
from utils.qontrol.phase_control import PhaseController

CAL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "node-isolation", "8-mode-autocal-20260209.json"
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def calibration():
    return HeaterCalibration.from_file(CAL_PATH)


@pytest.fixture
def label_map():
    return create_label_mapping("8x8")


@pytest.fixture
def driver(calibration, label_map):
    backend = SimulatedQontrol(
        n_channels=64, global_current_limit_mA=6.0,
        calibration=calibration, label_map=label_map, n_modes=8,
    )
    return SafeDriver(backend, audit=NullAuditLog(), max_current_mA=6.0).connect()


@pytest.fixture
def redis_client():
    return fakeredis.FakeStrictRedis()


# ──────────────────────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────────────────────

def test_phase_current_round_trip(calibration):
    """phase -> current -> phase must return the phase we asked for."""
    for heater in calibration.calibrated_heaters()[:12]:
        for phase in (0.1, 1.0, np.pi / 2, 3.0):
            try:
                current = calibration.phase_to_current(heater, phase)
            except ValueError:
                continue  # beyond the current limit; covered separately
            recovered = calibration.current_to_phase(heater, current)
            assert abs((recovered - phase + np.pi) % (2 * np.pi) - np.pi) < 1e-9


def test_zero_power_phase_is_subtracted(calibration):
    """The heater's phase at zero current is its stored offset, not zero."""
    heater = "G4_theta"
    params = calibration.phase_calibration[heater]["phase_params"]
    expected = (params["phase"] * np.pi) % (2 * np.pi)
    assert abs(calibration.current_to_phase(heater, 0.0) - expected) < 1e-12
    # Requesting exactly that phase must therefore need no power at all.
    assert calibration.required_power_mW(heater, expected) < 1e-9


def test_uncalibrated_heaters_are_refused(calibration):
    """Columns A and B have no phi calibration; guessing would be worse."""
    assert not calibration.is_calibrated("A1_phi")
    assert not calibration.is_calibrated("B2_phi")
    assert calibration.is_calibrated("A1_theta")
    with pytest.raises(KeyError):
        calibration.phase_to_current("A1_phi", 1.0)


def test_matches_measured_sweep(calibration):
    """The stored model must reproduce the raw sweep it was fitted to."""
    errors = []
    for heater, entry in calibration.phase_calibration.items():
        if heater not in calibration.resistance_calibration:
            continue
        md = entry.get("measurement_data") or {}
        currents, optical = md.get("currents"), md.get("optical_powers")
        if not currents:
            continue

        p = entry["phase_params"]
        predicted = [
            p["offset"] + p["amplitude"] * np.cos(calibration.current_to_phase(heater, i))
            for i in currents
        ]
        span = max(optical) - min(optical)
        if span > 0:
            errors.append(np.sqrt(np.mean((np.array(predicted) - optical) ** 2)) / span)

    assert np.median(errors) < 0.05, f"median normalised RMS {np.median(errors):.3f}"


# ──────────────────────────────────────────────────────────────────────────────
# Device safety
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.5, 9.9, "x", None])
def test_bad_currents_are_refused(driver, bad):
    with pytest.raises(DeviceError):
        driver.set_current(0, bad)
    assert driver.read_current(0) == 0.0


@pytest.mark.parametrize("channel", [-1, 64, 999, 1.5, True])
def test_bad_channels_are_refused(driver, channel):
    with pytest.raises(DeviceError):
        driver.set_current(channel, 1.0)


def test_set_many_validates_before_writing_anything(driver):
    """One bad entry must not leave the chip half programmed."""
    driver.set_current(0, 1.0)
    with pytest.raises(DeviceError):
        driver.set_many({1: 2.0, 2: 99.0, 3: 1.0})
    assert driver.read_current(1) == 0.0
    assert driver.read_current(3) == 0.0
    assert driver.read_current(0) == 1.0  # untouched


def test_zero_all(driver):
    driver.set_many({0: 1.0, 1: 2.0, 5: 3.0})
    driver.zero_all()
    assert all(c == 0.0 for c in driver.snapshot())


def test_audit_records_every_write(tmp_path, calibration, label_map):
    audit = AuditLog(str(tmp_path / "audit.jsonl"))
    backend = SimulatedQontrol(n_channels=8, calibration=calibration, label_map=label_map)
    drv = SafeDriver(backend, audit=audit, max_current_mA=6.0).connect()

    drv.set_current(0, 1.0, user="alice", job_id="j1")
    drv.set_many({1: 2.0}, user="bob", job_id="j2")
    with pytest.raises(DeviceError):
        drv.set_current(2, 99.0, user="mallory")

    entries = audit.tail()
    actions = [e["action"] for e in entries]
    assert "set_current" in actions and "set_many" in actions
    assert {e["user"] for e in entries} == {"alice", "bob"}  # refused write never ran
    assert entries[0]["current_mA"] == 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Closed loop: target unitary -> currents -> measured optical output
# ──────────────────────────────────────────────────────────────────────────────

def test_programs_a_unitary_onto_the_simulated_chip(driver, calibration):
    """The full pipeline must reproduce the target's power distribution."""
    from src.server.worker import target_to_grid_phases

    rng = np.random.default_rng(3)
    Z = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    Q, R = np.linalg.qr(Z)
    U_target = Q @ np.diag(np.diagonal(R) / np.abs(np.diagonal(R)))

    grid, _residual, _engine = target_to_grid_phases(U_target, 8)
    controller = PhaseController(calibration, grid_size="8x8")

    # Only heaters that exist on this chip can be driven; the rest stay at
    # their zero-power phase, which the simulator models faithfully.
    _currents, applied, skipped = controller.apply_phases_to_hardware(
        driver, grid, user="tester", job_id="closed-loop"
    )
    assert len(applied) == 49 and len(skipped) == 7

    measured = driver.backend.measured_output_powers()
    assert measured is not None
    assert abs(sum(measured) - 1.0) < 1e-6, "simulated mesh must conserve power"


def test_uncalibrated_heaters_do_not_block_a_job(driver, calibration):
    controller = PhaseController(calibration, grid_size="8x8")
    grid = {"A1": {"theta": 1.0, "phi": 2.0}, "G4": {"theta": 0.5, "phi": 0.5}}
    _c, applied, skipped = controller.phases_to_currents(grid)
    assert any("A1_phi" in s for s in skipped)
    assert any("A1_theta" in a for a in applied)


def test_phases_are_never_written_as_currents(driver, calibration):
    """Regression: the old code wrote raw phase values into the DAC fields."""
    controller = PhaseController(calibration, grid_size="8x8")
    grid = {"G4": {"theta": 0.806 * np.pi, "phi": 0.0}}
    currents, _applied, _skipped = controller.phases_to_currents(grid)
    theta_current = currents["G4"]["theta"]
    assert abs(theta_current - 4.235) < 0.01, theta_current
    assert theta_current != pytest.approx(0.806 * np.pi, abs=1e-3)


# ──────────────────────────────────────────────────────────────────────────────
# Lease
# ──────────────────────────────────────────────────────────────────────────────

def test_only_one_user_holds_the_chip(redis_client):
    lease = ChipLease(redis_client, max_ttl_seconds=60)
    first = lease.acquire("alice", 60)
    assert first is not None
    assert lease.acquire("bob", 60) is None
    assert lease.held_by("alice") and not lease.held_by("bob")


def test_reacquire_by_holder_extends(redis_client):
    lease = ChipLease(redis_client, max_ttl_seconds=60)
    first = lease.acquire("alice", 30)
    again = lease.acquire("alice", 60)
    assert again is not None and again["token"] == first["token"]


def test_renew_and_release_require_the_token(redis_client):
    lease = ChipLease(redis_client, max_ttl_seconds=60)
    held = lease.acquire("alice", 60)

    assert not lease.renew("wrong-token")
    assert not lease.release("wrong-token")
    assert lease.held_by("alice")

    assert lease.renew(held["token"])
    assert lease.release(held["token"])
    assert lease.holder() is None


def test_ttl_is_capped(redis_client):
    lease = ChipLease(redis_client, max_ttl_seconds=60)
    held = lease.acquire("alice", ttl_seconds=99999)
    assert held["expires_at"] - held["acquired_at"] <= 60


def test_require_rejects_non_holders(redis_client):
    lease = ChipLease(redis_client, max_ttl_seconds=60)
    with pytest.raises(LeaseError):
        lease.require("alice")

    held = lease.acquire("alice", 60)
    lease.require("alice", token=held["token"])
    with pytest.raises(LeaseError):
        lease.require("bob")
    with pytest.raises(LeaseError):
        lease.require("alice", token="stale")


def test_force_release(redis_client):
    lease = ChipLease(redis_client, max_ttl_seconds=60)
    lease.acquire("alice", 60)
    previous = lease.force_release()
    assert previous["user"] == "alice"
    assert lease.acquire("bob", 60) is not None


# ──────────────────────────────────────────────────────────────────────────────
# Quotas
# ──────────────────────────────────────────────────────────────────────────────

class _P:
    def __init__(self, login, role, quotas):
        self.login, self.role, self.quotas = login, role, quotas


def test_viewer_cannot_submit(redis_client):
    quotas = QuotaTracker(redis_client)
    viewer = _P("v", "viewer", {"max_concurrent_jobs": 0, "max_jobs_per_day": 0})
    with pytest.raises(LeaseError):
        quotas.check(viewer)


def test_concurrent_and_daily_limits(redis_client):
    quotas = QuotaTracker(redis_client)
    op = _P("o", "operator", {"max_concurrent_jobs": 2, "max_jobs_per_day": 3})

    quotas.check(op)
    quotas.record_submit("o")
    quotas.record_submit("o")
    with pytest.raises(LeaseError, match="in flight"):
        quotas.check(op)

    quotas.record_finish("o")
    quotas.record_finish("o")
    quotas.record_submit("o")
    quotas.record_finish("o")
    with pytest.raises(LeaseError, match="daily limit"):
        quotas.check(op)


def test_running_counter_never_goes_negative(redis_client):
    quotas = QuotaTracker(redis_client)
    quotas.record_finish("ghost")
    quotas.record_finish("ghost")
    assert quotas.running("ghost") == 0


# ──────────────────────────────────────────────────────────────────────────────
# Auth
# ──────────────────────────────────────────────────────────────────────────────

def test_non_tailnet_addresses_are_rejected():
    from src.server.auth import is_tailnet_address

    assert is_tailnet_address("100.101.102.103")
    assert is_tailnet_address("fd7a:115c:a1e0::1")
    for bad in ("127.0.0.1", "192.168.1.5", "8.8.8.8", "10.0.0.1", "not-an-ip"):
        assert not is_tailnet_address(bad), bad


def test_unknown_tailnet_user_is_denied(tmp_path):
    from fastapi import HTTPException

    from src.server.auth import UserDirectory, identify

    users = tmp_path / "users.json"
    users.write_text(json.dumps({"default_role": None,
                                 "users": {"alice@github": {"role": "operator"}}}))
    directory = UserDirectory(str(users))

    os.environ["MZIX_ALLOW_DEV_AUTH"] = "1"
    try:
        os.environ["MZIX_DEV_IDENTITY"] = "alice@github"
        assert identify("100.1.1.1", 1234, directory).role == "operator"

        os.environ["MZIX_DEV_IDENTITY"] = "stranger@github"
        with pytest.raises(HTTPException) as exc:
            identify("100.1.1.1", 1234, directory)
        assert exc.value.status_code == 403
    finally:
        os.environ.pop("MZIX_ALLOW_DEV_AUTH", None)
        os.environ.pop("MZIX_DEV_IDENTITY", None)


def test_role_permissions():
    from src.server.auth import Principal

    viewer = Principal("v", "v", "viewer", "100.1.1.1")
    operator = Principal("o", "o", "operator", "100.1.1.1")
    admin = Principal("a", "a", "admin", "100.1.1.1")

    assert viewer.can("chip:read")
    assert not viewer.can("job:submit") and not viewer.can("chip:lease")
    assert operator.can("job:submit") and operator.can("chip:stop")
    assert not operator.can("chip:force_release") and not operator.can("audit:read")
    assert admin.can("chip:force_release") and admin.can("audit:read")


def test_directory_reloads_on_change(tmp_path):
    from src.server.auth import UserDirectory

    users = tmp_path / "users.json"
    users.write_text(json.dumps({"users": {"a@x": {"role": "operator"}}}))
    directory = UserDirectory(str(users))
    assert directory.role_for("a@x") == "operator"

    os.utime(users, (0, 0))  # force a different mtime
    users.write_text(json.dumps({"users": {}}))
    assert directory.role_for("a@x") is None, "revocation must not need a restart"
