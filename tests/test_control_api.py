"""End-to-end API tests: identity, roles, lease arbitration, quotas.

Runs the real FastAPI app against fakeredis and the simulated chip. Identity
is supplied through the dev-auth escape hatch, which stands in for the
Tailscale whois lookup; everything downstream of identity is the real code.
"""

import json
import os
import sys

import fakeredis
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def api(tmp_path, monkeypatch):
    users = tmp_path / "users.json"
    users.write_text(json.dumps({
        "default_role": None,
        "users": {
            "boss@github": {"role": "admin"},
            "alice@github": {"role": "operator"},
            "bob@github": {"role": "operator"},
            "student@github": {"role": "viewer"},
        },
    }))

    monkeypatch.setenv("MZIX_ALLOW_DEV_AUTH", "1")
    monkeypatch.setenv("MZIX_USERS_FILE", str(users))
    monkeypatch.setenv("MZIX_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("MZIX_BACKEND", "simulator")
    monkeypatch.setenv("MZIX_MAX_LEASE_SECONDS", "60")

    import src.server.runtime as runtime

    _FACTORIES = ("get_settings", "get_redis", "get_chip_lease", "get_quota_tracker",
                  "get_audit_log", "get_calibration", "get_phase_controller",
                  "get_driver")

    def reset_runtime():
        # get_redis is monkeypatched to a plain callable during the test, so
        # only clear the ones that are still lru_cache wrappers.
        for name in _FACTORIES:
            factory = getattr(runtime, name)
            if hasattr(factory, "cache_clear"):
                factory.cache_clear()

    reset_runtime()
    shared_redis = fakeredis.FakeStrictRedis()
    monkeypatch.setattr(runtime, "get_redis", lambda: shared_redis)

    import src.server.main as main
    import src.server.worker as worker

    class _Queued:
        id = "task-123"

    monkeypatch.setattr(worker.process_batch_job, "apply_async",
                        lambda *a, **kw: _Queued())
    monkeypatch.setattr(worker.emergency_stop, "apply_async",
                        lambda *a, **kw: _Queued())

    with TestClient(main.app) as client:
        yield client

    reset_runtime()


def as_user(monkeypatch, login):
    monkeypatch.setenv("MZIX_DEV_IDENTITY", login)


def unitary_payload(name="test"):
    import numpy as np

    rng = np.random.default_rng(1)
    Z = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    Q, R = np.linalg.qr(Z)
    U = Q @ np.diag(np.diagonal(R) / np.abs(np.diagonal(R)))
    return {
        "unitaries": [{
            "name": name,
            "matrix_real": U.real.tolist(),
            "matrix_imag": U.imag.tolist(),
        }]
    }


# ──────────────────────────────────────────────────────────────────────────────
# Identity
# ──────────────────────────────────────────────────────────────────────────────

def test_identity_and_role(api, monkeypatch):
    as_user(monkeypatch, "alice@github")
    body = api.get("/api/v1/me").json()
    assert body["login"] == "alice@github"
    assert body["role"] == "operator"
    assert "job:submit" in body["permissions"]
    assert "chip:force_release" not in body["permissions"]


def test_unknown_user_is_denied(api, monkeypatch):
    as_user(monkeypatch, "intruder@github")
    assert api.get("/api/v1/me").status_code == 403


def test_unauthenticated_is_denied(api, monkeypatch):
    monkeypatch.delenv("MZIX_DEV_IDENTITY", raising=False)
    # No dev identity and the test client is not a tailnet address.
    assert api.get("/api/v1/me").status_code == 403


# ──────────────────────────────────────────────────────────────────────────────
# Lease arbitration
# ──────────────────────────────────────────────────────────────────────────────

def test_second_user_cannot_take_a_held_chip(api, monkeypatch):
    as_user(monkeypatch, "alice@github")
    assert api.post("/api/v1/chip/lease", json={}).status_code == 200

    as_user(monkeypatch, "bob@github")
    conflict = api.post("/api/v1/chip/lease", json={})
    assert conflict.status_code == 409
    assert "alice@github" in conflict.json()["detail"]


def test_viewer_cannot_lease_or_submit(api, monkeypatch):
    as_user(monkeypatch, "student@github")
    assert api.post("/api/v1/chip/lease", json={}).status_code == 403
    assert api.post("/api/v1/jobs", json=unitary_payload()).status_code == 403
    # but may still watch
    assert api.get("/api/v1/chip/status").status_code == 200


def test_lease_token_is_not_exposed_to_onlookers(api, monkeypatch):
    as_user(monkeypatch, "alice@github")
    token = api.post("/api/v1/chip/lease", json={}).json()["token"]

    as_user(monkeypatch, "bob@github")
    lease = api.get("/api/v1/chip/status").json()["lease"]
    assert lease["user"] == "alice@github"
    assert "token" not in lease and token not in json.dumps(lease)


def test_release_requires_the_right_token(api, monkeypatch):
    as_user(monkeypatch, "alice@github")
    token = api.post("/api/v1/chip/lease", json={}).json()["token"]

    as_user(monkeypatch, "bob@github")
    assert api.request("DELETE", "/api/v1/chip/lease",
                       json={"token": "guessed"}).status_code == 409

    as_user(monkeypatch, "alice@github")
    assert api.request("DELETE", "/api/v1/chip/lease",
                       json={"token": token}).status_code == 200


def test_only_admin_can_force_release(api, monkeypatch):
    as_user(monkeypatch, "alice@github")
    api.post("/api/v1/chip/lease", json={})

    as_user(monkeypatch, "bob@github")
    assert api.delete("/api/v1/chip/lease/force").status_code == 403

    as_user(monkeypatch, "boss@github")
    forced = api.delete("/api/v1/chip/lease/force")
    assert forced.status_code == 200
    assert forced.json()["previous"]["user"] == "alice@github"

    as_user(monkeypatch, "bob@github")
    assert api.post("/api/v1/chip/lease", json={}).status_code == 200


def test_lease_ttl_is_capped_by_role_quota(api, monkeypatch):
    as_user(monkeypatch, "alice@github")
    lease = api.post("/api/v1/chip/lease", json={"ttl_seconds": 999999}).json()
    assert lease["expires_at"] - lease["acquired_at"] <= 60


# ──────────────────────────────────────────────────────────────────────────────
# Jobs
# ──────────────────────────────────────────────────────────────────────────────

def test_job_requires_holding_the_lease(api, monkeypatch):
    as_user(monkeypatch, "alice@github")
    refused = api.post("/api/v1/jobs", json=unitary_payload())
    assert refused.status_code == 409
    assert "lease" in refused.json()["detail"]

    api.post("/api/v1/chip/lease", json={})
    accepted = api.post("/api/v1/jobs", json=unitary_payload())
    assert accepted.status_code == 200
    assert accepted.json()["status"] == "queued"


def test_job_cannot_be_submitted_on_someone_elses_lease(api, monkeypatch):
    as_user(monkeypatch, "alice@github")
    api.post("/api/v1/chip/lease", json={})

    as_user(monkeypatch, "bob@github")
    refused = api.post("/api/v1/jobs", json=unitary_payload())
    assert refused.status_code == 409


def test_identity_cannot_be_forged_through_the_payload(api, monkeypatch):
    """user_id used to come from the body; it must now be ignored."""
    as_user(monkeypatch, "alice@github")
    api.post("/api/v1/chip/lease", json={})

    payload = unitary_payload()
    payload["user_id"] = "boss@github"
    assert api.post("/api/v1/jobs", json=payload).status_code == 200

    as_user(monkeypatch, "boss@github")
    entries = api.get("/api/v1/audit").json()["entries"]
    queued = [e for e in entries if e["action"] == "job_queued"]
    assert queued and all(e["user"] == "alice@github" for e in queued)


def test_concurrent_job_quota_is_enforced(api, monkeypatch):
    as_user(monkeypatch, "alice@github")
    api.post("/api/v1/chip/lease", json={})

    assert api.post("/api/v1/jobs", json=unitary_payload("a")).status_code == 200
    assert api.post("/api/v1/jobs", json=unitary_payload("b")).status_code == 200
    throttled = api.post("/api/v1/jobs", json=unitary_payload("c"))
    assert throttled.status_code == 429
    assert "in flight" in throttled.json()["detail"]


def test_malformed_targets_are_rejected(api, monkeypatch):
    as_user(monkeypatch, "alice@github")
    api.post("/api/v1/chip/lease", json={})

    bad = {"unitaries": [{"name": "ragged",
                          "matrix_real": [[1.0, 0.0], [0.0]],
                          "matrix_imag": [[0.0, 0.0], [0.0, 0.0]]}]}
    assert api.post("/api/v1/jobs", json=bad).status_code == 422

    empty = {"unitaries": []}
    assert api.post("/api/v1/jobs", json=empty).status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# Audit
# ──────────────────────────────────────────────────────────────────────────────

def test_audit_is_admin_only(api, monkeypatch):
    as_user(monkeypatch, "alice@github")
    assert api.get("/api/v1/audit").status_code == 403

    as_user(monkeypatch, "boss@github")
    assert api.get("/api/v1/audit").status_code == 200


def test_audit_records_lease_and_job_activity(api, monkeypatch):
    as_user(monkeypatch, "alice@github")
    api.post("/api/v1/chip/lease", json={})
    api.post("/api/v1/jobs", json=unitary_payload())

    as_user(monkeypatch, "boss@github")
    actions = [e["action"] for e in api.get("/api/v1/audit").json()["entries"]]
    assert "lease_acquired" in actions and "job_queued" in actions
