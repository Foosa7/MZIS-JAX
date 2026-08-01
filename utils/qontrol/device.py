"""Current-driver backends, with the safety envelope in one place.

Everything that can physically damage the chip lives here: current limits,
NaN rejection, and the emergency zero. `SafeDriver` wraps any backend and is
the only thing the rest of the stack is allowed to talk to, so a new backend
cannot accidentally opt out of the checks.

`SimulatedQontrol` is not a stub that swallows writes -- it runs the currents
back through the calibration and the JAX mesh, so the whole server can be
exercised end to end, closed loop, with no chip attached.
"""

import logging
import math
import threading


class DeviceError(RuntimeError):
    """Raised when a backend refuses or fails a write."""


# ──────────────────────────────────────────────────────────────────────────────
# Backends
# ──────────────────────────────────────────────────────────────────────────────

class QontrolBackend:
    """Interface every current driver must provide.

    Channels are zero-indexed integers, currents are milliamps.
    """

    n_channels = 0
    global_current_limit_mA = 0.0

    def connect(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    @property
    def connected(self):
        raise NotImplementedError

    def write_current(self, channel, current_mA):
        raise NotImplementedError

    def read_current(self, channel):
        raise NotImplementedError


class SimulatedQontrol(QontrolBackend):
    """In-process chip: currents -> phases -> mesh unitary -> output powers.

    Optional, because the server must still start when JAX is unavailable;
    without an engine it behaves as a pure current register.
    """

    def __init__(self, n_channels=64, global_current_limit_mA=6.0,
                 calibration=None, label_map=None, n_modes=8):
        self.n_channels = n_channels
        self.global_current_limit_mA = global_current_limit_mA
        self.calibration = calibration
        self.label_map = label_map or {}
        self.n_modes = n_modes

        self._currents = [0.0] * n_channels
        self._connected = False
        self._engine = None

    def connect(self):
        self._connected = True
        if self.calibration is not None:
            try:
                self._build_engine()
            except Exception as exc:  # simulator stays usable without JAX
                logging.warning("simulator running without optical model: %s", exc)
        return self

    def _build_engine(self):
        from src.engine import Engine

        engine = Engine(n_modes=self.n_modes)
        errors = [self.calibration.beamsplitter_error(f"{mid}_theta")
                  for mid in engine.mzi_ids]
        engine.set_bs_error(tuple([errors, errors]))
        self._engine = engine

    def close(self):
        self._connected = False

    @property
    def connected(self):
        return self._connected

    def write_current(self, channel, current_mA):
        if not self._connected:
            raise DeviceError("simulated device is not connected")
        self._currents[channel] = current_mA

    def read_current(self, channel):
        return self._currents[channel]

    # ── optical model ────────────────────────────────────────────────────

    def measured_phases(self):
        """Phase every calibrated heater currently sits at, keyed by heater id."""
        if self.calibration is None:
            return {}
        phases = {}
        for label, (theta_ch, phi_ch) in self.label_map.items():
            for arm, channel in (("theta", theta_ch), ("phi", phi_ch)):
                heater = f"{label}_{arm}"
                if self.calibration.is_calibrated(heater) and channel < self.n_channels:
                    phases[heater] = self.calibration.current_to_phase(
                        heater, self._currents[channel]
                    )
        return phases

    def measured_output_powers(self, input_powers=None):
        """Output power per port for the currents presently loaded.

        Returns None when no optical model is available.
        """
        if self._engine is None:
            return None

        import numpy as np
        import jax.numpy as jnp

        phases = self.measured_phases()
        thetas, phis = [], []
        for mid in self._engine.mzi_ids:
            thetas.append(phases.get(f"{mid}_theta", 0.0))
            phis.append(phases.get(f"{mid}_phi", 0.0))

        U = np.asarray(self._engine.compute_full_unitary(
            jnp.array(thetas), jnp.array(phis)
        ))
        if input_powers is None:
            input_powers = np.zeros(self.n_modes)
            input_powers[0] = 1.0
        return (np.abs(U) ** 2 @ np.asarray(input_powers, dtype=float)).tolist()


class RealQontrol(QontrolBackend):
    """Thin adapter over the `qontrol` package's QXOutput."""

    def __init__(self, serial_port=None, global_current_limit_mA=6.0):
        self.serial_port = serial_port
        self.global_current_limit_mA = global_current_limit_mA
        self._device = None

    def connect(self):
        try:
            import qontrol
        except ImportError as exc:
            raise DeviceError(
                "the `qontrol` package is required for hardware mode "
                "(pip install qontrol)"
            ) from exc

        self._device = qontrol.QXOutput(serial_port_name=self.serial_port)
        self.n_channels = int(getattr(self._device, 'n_chs', 0) or 0)

        # Trust the device's own limit when it reports one lower than ours.
        device_limit = getattr(self._device, 'globalcurrentlimit', None)
        if device_limit:
            self.global_current_limit_mA = min(self.global_current_limit_mA,
                                               float(device_limit))
        return self

    def close(self):
        if self._device is not None:
            try:
                self._device.close()
            finally:
                self._device = None

    @property
    def connected(self):
        return self._device is not None

    def write_current(self, channel, current_mA):
        if self._device is None:
            raise DeviceError("Qontrol device is not connected")
        self._device.i[channel] = current_mA

    def read_current(self, channel):
        if self._device is None:
            raise DeviceError("Qontrol device is not connected")
        return float(self._device.i[channel])


# ──────────────────────────────────────────────────────────────────────────────
# Safety wrapper
# ──────────────────────────────────────────────────────────────────────────────

class SafeDriver:
    """Serialises access to a backend and enforces the current envelope.

    Every write is validated and, if an audit sink is supplied, recorded
    before it reaches the device -- so the log cannot omit a write that
    actually happened.
    """

    def __init__(self, backend, audit=None, max_current_mA=None):
        self.backend = backend
        self.audit = audit
        self.max_current_mA = (
            max_current_mA
            if max_current_mA is not None
            else backend.global_current_limit_mA
        )
        self._lock = threading.Lock()

    # ── lifecycle ────────────────────────────────────────────────────────

    def connect(self):
        with self._lock:
            self.backend.connect()
        return self

    def close(self):
        with self._lock:
            self.backend.close()

    @property
    def connected(self):
        return self.backend.connected

    @property
    def n_channels(self):
        return self.backend.n_channels

    # ── validation ───────────────────────────────────────────────────────

    def _validate(self, channel, current_mA):
        if not isinstance(channel, int) or isinstance(channel, bool):
            raise DeviceError(f"channel must be an int, got {channel!r}")
        if not 0 <= channel < self.backend.n_channels:
            raise DeviceError(
                f"channel {channel} outside 0..{self.backend.n_channels - 1}"
            )

        try:
            current = float(current_mA)
        except (TypeError, ValueError) as exc:
            raise DeviceError(f"current must be a number, got {current_mA!r}") from exc

        if math.isnan(current) or math.isinf(current):
            raise DeviceError(f"refusing non-finite current on channel {channel}")
        if current < 0:
            raise DeviceError(
                f"refusing negative current {current} mA on channel {channel}"
            )
        if current > self.max_current_mA:
            raise DeviceError(
                f"current {current:.4f} mA on channel {channel} exceeds the "
                f"{self.max_current_mA} mA limit"
            )
        return current

    # ── writes ───────────────────────────────────────────────────────────

    def set_current(self, channel, current_mA, user=None, job_id=None):
        current = self._validate(channel, current_mA)
        with self._lock:
            if self.audit is not None:
                self.audit.record("set_current", user=user, job_id=job_id,
                                  channel=channel, current_mA=current)
            self.backend.write_current(channel, current)

    def set_many(self, channel_currents, user=None, job_id=None):
        """Applies a {channel: mA} batch atomically with respect to validation.

        Every value is checked before any of them is written, so a bad entry
        cannot leave the chip in a half-programmed state.
        """
        validated = {ch: self._validate(ch, mA) for ch, mA in channel_currents.items()}
        with self._lock:
            if self.audit is not None:
                self.audit.record("set_many", user=user, job_id=job_id,
                                  channels=len(validated),
                                  total_current_mA=round(sum(validated.values()), 6))
            for channel, current in sorted(validated.items()):
                self.backend.write_current(channel, current)
        return validated

    def zero_all(self, user=None, reason="zero_all"):
        """Drops every channel to 0 mA. The emergency path -- never blocks."""
        with self._lock:
            if self.audit is not None:
                self.audit.record(reason, user=user, channels=self.backend.n_channels)
            errors = []
            for channel in range(self.backend.n_channels):
                try:
                    self.backend.write_current(channel, 0.0)
                except Exception as exc:
                    errors.append((channel, str(exc)))
            if errors:
                raise DeviceError(f"failed to zero {len(errors)} channel(s): {errors[:3]}")

    def read_current(self, channel):
        with self._lock:
            return self.backend.read_current(channel)

    def snapshot(self):
        with self._lock:
            return [self.backend.read_current(c) for c in range(self.backend.n_channels)]
