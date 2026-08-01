"""Identity and authorisation, sourced from Tailscale.

The previous "handshake" GET a page on the server and, if it returned 200,
declared the session authenticated -- the password was never transmitted and
the API had no auth at all. This replaces it with the tailnet's own identity.

The caller's tailnet address is resolved through the local tailscaled API
(`whois`), which is authoritative: the address is assigned by the coordination
server and cannot be spoofed by a peer, and only nodes your ACLs admit can
open the connection at all. There are no passwords to store here as a result.

Two invariants keep that guarantee honest:

* the peer address must sit inside the Tailscale range, so a request arriving
  over localhost or the LAN cannot borrow an identity; and
* forwarded-for headers are never consulted. Run this bound to the tailnet
  address with no reverse proxy in front.
"""

import ipaddress
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Depends, HTTPException, Request, status

log = logging.getLogger("mzix.auth")

# Tailscale hands out addresses from the CGNAT block and one ULA prefix.
_TAILNET_V4 = ipaddress.ip_network("100.64.0.0/10")
_TAILNET_V6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")

_LOCALAPI_SOCKETS = (
    "/var/run/tailscale/tailscaled.sock",
    "/run/tailscale/tailscaled.sock",
)

# ──────────────────────────────────────────────────────────────────────────────
# Roles
# ──────────────────────────────────────────────────────────────────────────────

PERMISSIONS = {
    "viewer": {"chip:read"},
    "operator": {"chip:read", "chip:lease", "chip:stop", "job:submit", "job:cancel_own"},
    "admin": {"chip:read", "chip:lease", "chip:stop", "job:submit", "job:cancel_own",
              "chip:force_release", "job:cancel_any", "users:read", "audit:read"},
}

DEFAULT_QUOTAS = {
    "viewer": {"max_concurrent_jobs": 0, "max_lease_seconds": 0, "max_jobs_per_day": 0},
    "operator": {"max_concurrent_jobs": 2, "max_lease_seconds": 1800, "max_jobs_per_day": 200},
    "admin": {"max_concurrent_jobs": 8, "max_lease_seconds": 7200, "max_jobs_per_day": 2000},
}


@dataclass(frozen=True)
class Principal:
    """An authenticated caller."""

    login: str
    display_name: str
    role: str
    address: str
    quotas: dict = field(default_factory=dict)

    def can(self, permission):
        return permission in PERMISSIONS.get(self.role, set())


# ──────────────────────────────────────────────────────────────────────────────
# User directory
# ──────────────────────────────────────────────────────────────────────────────

class UserDirectory:
    """Maps a Tailscale login to a role, from a JSON file.

    Reloaded when the file changes, so access can be revoked without a
    restart. Unknown users are denied unless `default_role` is set.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._mtime = None
        self._data = {"users": {}, "default_role": None, "quotas": {}}
        self.reload()

    def reload(self):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            log.warning("user directory %s not found; denying all callers", self.path)
            self._data = {"users": {}, "default_role": None, "quotas": {}}
            self._mtime = None
            return

        if self._mtime == stat.st_mtime:
            return
        with open(self.path) as f:
            self._data = json.load(f)
        self._mtime = stat.st_mtime
        log.info("loaded %d user(s) from %s", len(self._data.get("users", {})), self.path)

    def role_for(self, login):
        self.reload()
        users = self._data.get("users", {})
        entry = users.get(login)
        if isinstance(entry, dict):
            role = entry.get("role")
        else:
            role = entry
        role = role or self._data.get("default_role")
        if role is not None and role not in PERMISSIONS:
            log.error("user %s has unknown role %r; denying", login, role)
            return None
        return role

    def quotas_for(self, role):
        self.reload()
        quotas = dict(DEFAULT_QUOTAS.get(role, DEFAULT_QUOTAS["viewer"]))
        quotas.update(self._data.get("quotas", {}).get(role, {}))
        return quotas

    def all_users(self):
        self.reload()
        return dict(self._data.get("users", {}))


# ──────────────────────────────────────────────────────────────────────────────
# Tailscale whois
# ──────────────────────────────────────────────────────────────────────────────

def is_tailnet_address(host):
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return addr in _TAILNET_V4 or addr in _TAILNET_V6


def _whois_localapi(addr):
    """Queries tailscaled's local API over its unix socket."""
    import httpx

    socket_path = next((p for p in _LOCALAPI_SOCKETS if os.path.exists(p)), None)
    if socket_path is None:
        return None

    try:
        transport = httpx.HTTPTransport(uds=socket_path)
        with httpx.Client(transport=transport, timeout=3.0) as client:
            resp = client.get(
                f"http://local-tailscaled.sock/localapi/v0/whois?addr={addr}",
                headers={"Host": "local-tailscaled.sock"},
            )
        if resp.status_code != 200:
            log.warning("tailscale whois returned %s for %s", resp.status_code, addr)
            return None
        return resp.json()
    except Exception as exc:
        log.warning("tailscale local API unavailable (%s); falling back to CLI", exc)
        return None


def _whois_cli(addr):
    """Falls back to the tailscale binary when the socket is not readable."""
    try:
        out = subprocess.run(
            ["tailscale", "whois", "--json", addr],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return json.loads(out.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        log.warning("tailscale whois CLI failed for %s: %s", addr, exc)
        return None


def whois(host, port):
    """Resolves a tailnet peer address to its user profile, or None."""
    addr = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
    result = _whois_localapi(addr) or _whois_cli(addr)
    if not result:
        return None

    profile = result.get("UserProfile") or {}
    login = profile.get("LoginName")
    if not login:
        return None
    return {
        "login": login,
        "display_name": profile.get("DisplayName") or login,
        "node": (result.get("Node") or {}).get("Name", ""),
    }


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI wiring
# ──────────────────────────────────────────────────────────────────────────────

def _dev_identity():
    """Escape hatch for running the stack without a tailnet.

    Deliberately noisy and off unless both variables are set, so it cannot be
    left switched on by accident in the lab.
    """
    if os.environ.get("MZIX_ALLOW_DEV_AUTH") != "1":
        return None
    login = os.environ.get("MZIX_DEV_IDENTITY")
    if not login:
        return None
    log.warning("MZIX_ALLOW_DEV_AUTH is set: trusting identity %r without Tailscale", login)
    return {"login": login, "display_name": login, "node": "dev"}


def get_user_directory(request: Request) -> UserDirectory:
    directory = getattr(request.app.state, "users", None)
    if directory is None:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            "user directory is not configured")
    return directory


def identify(client_host, client_port, directory):
    """Shared by the HTTP and WebSocket paths. Returns a Principal or raises."""
    identity = _dev_identity()

    if identity is None:
        if client_host is None or not is_tailnet_address(client_host):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"requests must arrive over the tailnet; got {client_host!r}",
            )
        identity = whois(client_host, client_port)
        if identity is None:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "could not resolve caller identity via Tailscale",
            )

    role = directory.role_for(identity["login"])
    if role is None:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"{identity['login']} is on the tailnet but not authorised for this chip",
        )

    return Principal(
        login=identity["login"],
        display_name=identity["display_name"],
        role=role,
        address=str(client_host),
        quotas=directory.quotas_for(role),
    )


async def current_user(request: Request,
                       directory: UserDirectory = Depends(get_user_directory)) -> Principal:
    client = request.client
    return identify(client.host if client else None,
                    client.port if client else 0,
                    directory)


def require(permission):
    """Dependency factory: rejects callers lacking `permission`."""

    async def _dependency(user: Principal = Depends(current_user)) -> Principal:
        if not user.can(permission):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"role {user.role!r} may not {permission}",
            )
        return user

    return _dependency
