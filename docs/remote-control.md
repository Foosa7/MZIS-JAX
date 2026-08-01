# Remote chip control

Running the 8-mode chip from anywhere, over Tailscale, with several people
sharing one bench.

## Shape of it

```
your laptop ──tailnet──► lab machine
                          ├── uvicorn  src.server.main:app   (identity, leases, queueing)
                          ├── celery   src.server.worker     (the only process that drives current)
                          └── redis                          (queue, lease, telemetry pub/sub)
```

The API never writes to the chip. It authenticates callers, arbitrates the
lease and queues work; the worker is the sole writer. That is what keeps one
writer per chip even with several API workers running.

## Security model

Access control is Tailscale's, not ours. The caller's tailnet address is
resolved through the local `tailscaled` API (`whois`), which is authoritative:
addresses are assigned by the coordination server and a peer cannot spoof one.
There are no passwords stored anywhere in this repo.

Two invariants make that hold, and both will bite you if ignored:

1. **Bind to the tailnet address, never `0.0.0.0`.** The server trusts the
   peer address, so anything else able to reach the port could be treated as
   a tailnet peer. `MZIX_BIND_HOST` defaults to loopback and the server logs
   an error if you set it to a wildcard.
2. **Do not put a reverse proxy in front.** Forwarded-for headers are never
   consulted; a proxy would make every caller look like the proxy.

Requests arriving from outside the Tailscale ranges (`100.64.0.0/10`,
`fd7a:115c:a1e0::/48`) are rejected before identity is considered.

Restrict who can reach the port at all with a tailnet ACL:

```jsonc
{
  "acls": [
    { "action": "accept",
      "src": ["group:photonics"],
      "dst": ["lab-machine:8000"] }
  ]
}
```

## Roles

Being on the tailnet is not authorisation. Users are listed in
`config/users.json` (copy `config/users.example.json`); anyone not listed is
denied unless you set `default_role`. The file is re-read when it changes, so
revoking access does not need a restart.

| | viewer | operator | admin |
|---|---|---|---|
| read chip status, own telemetry | ✅ | ✅ | ✅ |
| hold the chip lease | | ✅ | ✅ |
| submit jobs, emergency stop | | ✅ | ✅ |
| force-release someone's lease | | | ✅ |
| read the audit log, list users | | | ✅ |

Quotas are per role and overridable per role in `users.json`:
`max_concurrent_jobs`, `max_lease_seconds`, `max_jobs_per_day`.

## The lease

One chip, one writer. Before submitting work you take the lease:

```bash
TS=$(tailscale ip -4 lab-machine)
TOKEN=$(curl -s -X POST http://$TS:8000/api/v1/chip/lease \
          -H 'Content-Type: application/json' \
          -d '{"ttl_seconds": 900, "note": "sweeping unitaries"}' | jq -r .token)

curl -X POST http://$TS:8000/api/v1/jobs -H 'Content-Type: application/json' \
     -d @job.json

curl -X DELETE http://$TS:8000/api/v1/chip/lease \
     -H 'Content-Type: application/json' -d "{\"token\": \"$TOKEN\"}"
```

The lease carries a TTL, so a client that crashes frees the bench instead of
wedging it; renew with `POST /api/v1/chip/lease/renew` for long sessions. The
worker re-checks the lease before every write, so a lease that lapses
mid-job stops the next write rather than the next job. An admin can take a
lease back with `DELETE /api/v1/chip/lease/force`.

## Running it

```bash
pip install -r server_requirements.txt      # lab machine
cp config/users.example.json config/users.json && $EDITOR config/users.json

export MZIX_BIND_HOST=$(tailscale ip -4)
export MZIX_BACKEND=simulator               # switch to `qontrol` for real hardware
redis-server &
celery -A src.server.worker worker --loglevel=info --concurrency=1 &
uvicorn src.server.main:app --host $MZIX_BIND_HOST --port 8000
```

`--concurrency=1` is not a performance setting. Two workers interleave writes
to the same DACs.

### Environment

| variable | default | meaning |
|---|---|---|
| `MZIX_BACKEND` | `simulator` | `simulator` or `qontrol` |
| `MZIX_BIND_HOST` | `127.0.0.1` | set to the Tailscale address |
| `MZIX_SERIAL_PORT` | auto | Qontrol serial port |
| `MZIX_MAX_CURRENT_MA` | `6.0` | hard ceiling per channel |
| `MZIX_CALIBRATION` | 8-mode autocal | autocal JSON path |
| `MZIX_USERS_FILE` | `config/users.json` | role directory |
| `MZIX_AUDIT_LOG` | `var/audit.jsonl` | append-only hardware log |
| `MZIX_MAX_LEASE_SECONDS` | `1800` | ceiling on any lease |
| `MZIX_ALLOW_DEV_AUTH` | unset | **development only**, see below |

## Going to real hardware

`MZIX_BACKEND=qontrol` and `pip install qontrol`. Nothing else changes: the
simulator implements the same interface, and every write already goes through
`SafeDriver`, which rejects NaN, negative, over-limit and out-of-range
channels, and validates a whole batch before writing any of it.

Before trusting it on the bench:

- confirm `MZIX_MAX_CURRENT_MA` is at or below what the heaters tolerate;
- check `/api/v1/chip/status` lists the 7 uncalibrated heaters you expect
  (the φ heaters of columns A and B in the current autocal set) — those are
  skipped, not guessed at;
- verify `POST /api/v1/chip/emergency-stop` zeroes every channel.

## Development without a tailnet

Setting both `MZIX_ALLOW_DEV_AUTH=1` and `MZIX_DEV_IDENTITY=you@example.com`
makes the server trust that identity without Tailscale. It is off unless both
are set and logs a warning on every request. Never set it on the lab machine.

## Audit

Every hardware write is appended to `var/audit.jsonl` **before** it reaches
the device, so the log can contain an attempt that failed but never omit one
that succeeded. Admins can read it via `GET /api/v1/audit`, or just:

```bash
jq -r 'select(.action=="set_many") | "\(.iso) \(.user) \(.channels)ch \(.total_current_mA)mA"' var/audit.jsonl
```
