"""Per-heater conversion between optical phase and DAC current.

The node-isolation autocal file stores, for every heater, a fitted fringe

    I_optical = offset + amplitude * cos(omega * P + phase)

and a thermo-optic resistance model. Both have sharp edges that are easy to
get wrong, so they are pinned down here once:

* `phase` is the heater's phase at zero electrical power and is stored in
  **units of pi**, not radians. It must be subtracted from the request.
* The resistance model that reproduces the measured sweeps is
  `P = c_res * I^2 * (1 + alpha_res * I^2)`, not the c/a/d cubic.

Fitting each candidate back against `measurement_data` is unambiguous:
the model above with `phase` in units of pi gives a median normalised RMS
of 0.008 across all 49 heaters, the alternatives 0.43 to 0.49. Getting it
wrong is a ~38% current error, which is silent and lands on real hardware.
"""

import json
import math

# Heaters the mesh exposes but the autocal run never characterised (the
# external phases of the first two columns). Requests for these are refused
# rather than guessed at.
_UNCALIBRATED = object()


class HeaterCalibration:
    """Phase <-> current for one chip, backed by an autocal JSON file."""

    def __init__(self, cal_data, max_current_mA=6.0):
        self.phase_calibration = cal_data.get('phase_calibration', {})
        self.resistance_calibration = cal_data.get('resistance_calibration', {})
        self.metadata = cal_data.get('metadata', {})
        self.max_current_mA = max_current_mA

    @classmethod
    def from_file(cls, path, max_current_mA=6.0):
        with open(path, 'r') as f:
            return cls(json.load(f), max_current_mA=max_current_mA)

    # ──────────────────────────────────────────────────────────────────────
    # Introspection
    # ──────────────────────────────────────────────────────────────────────

    def is_calibrated(self, heater_id):
        """True if this heater can be driven to a requested phase."""
        return self._params(heater_id) is not _UNCALIBRATED

    def calibrated_heaters(self):
        return sorted(k for k in self.phase_calibration if self.is_calibrated(k))

    def _params(self, heater_id):
        phase_cal = self.phase_calibration.get(heater_id)
        res_cal = self.resistance_calibration.get(heater_id)
        if phase_cal is None or res_cal is None:
            return _UNCALIBRATED

        p = phase_cal.get('phase_params') or {}
        r = res_cal.get('resistance_params') or {}
        needed_p = ('omega', 'phase')
        needed_r = ('c_res', 'alpha_res')
        if any(p.get(k) is None for k in needed_p) or any(r.get(k) is None for k in needed_r):
            return _UNCALIBRATED
        if not p['omega'] or not r['c_res']:
            return _UNCALIBRATED
        return p, r

    # ──────────────────────────────────────────────────────────────────────
    # Phase -> current
    # ──────────────────────────────────────────────────────────────────────

    def required_power_mW(self, heater_id, phase_rad):
        """Electrical power needed to reach `phase_rad` on this heater.

        Heaters only add phase, so the request is wrapped into [0, 2*pi)
        relative to the heater's zero-power phase.
        """
        params = self._params(heater_id)
        if params is _UNCALIBRATED:
            raise KeyError(f"no usable calibration for heater {heater_id!r}")
        p, _ = params

        phase_at_zero_power = p['phase'] * math.pi
        delta = (phase_rad - phase_at_zero_power) % (2 * math.pi)
        return delta / p['omega']

    def phase_to_current(self, heater_id, phase_rad):
        """Current in mA that puts this heater at `phase_rad`.

        Inverts P = c*I^2*(1 + alpha*I^2), a quadratic in I^2:
            c*alpha*x^2 + c*x - P = 0
        """
        params = self._params(heater_id)
        if params is _UNCALIBRATED:
            raise KeyError(f"no usable calibration for heater {heater_id!r}")
        _, r = params

        power = self.required_power_mW(heater_id, phase_rad)
        c_res, alpha = r['c_res'], r['alpha_res']

        if alpha == 0:
            i_squared = power / c_res
        else:
            discriminant = 1.0 + 4.0 * alpha * (power / c_res)
            i_squared = (-1.0 + math.sqrt(max(0.0, discriminant))) / (2.0 * alpha)

        current = math.sqrt(max(0.0, i_squared))
        # The caller still clamps at the device, but refuse to hand back a
        # value we already know is unreachable rather than silently clipping.
        if current > self.max_current_mA:
            raise ValueError(
                f"heater {heater_id}: phase {phase_rad:.3f} rad needs "
                f"{current:.3f} mA, above the {self.max_current_mA} mA limit"
            )
        return current

    # ──────────────────────────────────────────────────────────────────────
    # Current -> phase (readback / warm start)
    # ──────────────────────────────────────────────────────────────────────

    def current_to_phase(self, heater_id, current_mA):
        """Phase this heater sits at when driven with `current_mA`."""
        params = self._params(heater_id)
        if params is _UNCALIBRATED:
            raise KeyError(f"no usable calibration for heater {heater_id!r}")
        p, r = params

        i_sq = current_mA ** 2
        power = r['c_res'] * i_sq * (1.0 + r['alpha_res'] * i_sq)
        return (p['omega'] * power + p['phase'] * math.pi) % (2 * math.pi)

    # ──────────────────────────────────────────────────────────────────────
    # Beamsplitter error, for the simulator's forward model
    # ──────────────────────────────────────────────────────────────────────

    def beamsplitter_error(self, heater_id, default=0.0):
        """Extracts epsilon from fringe visibility: eps^2 = (C - A) / (C + A).

        Nodes whose fitted amplitude meets or exceeds the offset carry no
        usable visibility information and fall back to `default`.
        """
        phase_cal = self.phase_calibration.get(heater_id)
        if not phase_cal:
            return default
        p = phase_cal.get('phase_params') or {}
        A, C = p.get('amplitude'), p.get('offset')
        if A is None or C is None or (C + A) <= 0:
            return default
        eps_sq = (C - A) / (C + A)
        return math.sqrt(eps_sq) if eps_sq > 0 else default
