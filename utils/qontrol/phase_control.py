"""Turning a grid of optical phases into currents on the chip.

Two things were wrong here before:

* the module imported `app.utils.appdata`, which does not exist in this
  repository, so it raised ImportError on load; and
* `calculate_current_for_phase` called `solve_current_with_brentq`, which is
  commented out. Every channel therefore raised AttributeError, and because
  `apply_phases_to_hardware` caught per-channel exceptions and carried on, the
  grid it handed to the device still held **phase values in the current
  fields** -- it would have driven the DACs with phases in place of milliamps.

Calibration is now injected rather than read from global state, and the
analytic solver is the only path.
"""

import copy
import logging

from utils.qontrol.mapping_utils import apply_channel_currents, create_label_mapping


class PhaseController:
    """Converts phases to currents for one chip, given its calibration."""

    def __init__(self, calibration, grid_size="8x8"):
        self.calibration = calibration
        self.grid_size = grid_size
        self.label_map = create_label_mapping(grid_size)

    # ──────────────────────────────────────────────────────────────────────
    # Single heater
    # ──────────────────────────────────────────────────────────────────────

    def calculate_current_for_phase(self, calib_key, phase_value_rad):
        """Current in mA for one heater, or None if it has no calibration.

        Uncalibrated heaters are a real feature of the 8-mode autocal set --
        the external phases of columns A and B were never characterised -- so
        they are skipped rather than treated as an error.
        """
        if not self.calibration.is_calibrated(calib_key):
            return None
        return self.calibration.phase_to_current(calib_key, phase_value_rad)

    # ──────────────────────────────────────────────────────────────────────
    # Whole grid
    # ──────────────────────────────────────────────────────────────────────

    def phases_to_currents(self, grid_config):
        """Converts {label: {'theta': rad, 'phi': rad}} into currents.

        Returns (grid_currents, applied, skipped). A heater that cannot be
        driven lands in `skipped`; anything genuinely wrong raises, so a
        half-converted grid never reaches the device.
        """
        grid_currents = {}
        applied, skipped = [], []

        for label, data in grid_config.items():
            if label not in self.label_map:
                continue

            entry = {}
            for arm in ("theta", "phi"):
                if arm not in data or data[arm] in (None, ""):
                    continue

                try:
                    phase = float(data[arm])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{label}.{arm} is {data[arm]!r}, expected a phase in radians"
                    ) from exc

                heater = f"{label}_{arm}"
                current = self.calculate_current_for_phase(heater, phase)
                if current is None:
                    skipped.append(f"{heater} (no calibration)")
                    continue

                entry[arm] = round(current, 5)
                applied.append(f"{heater} = {current:.5f} mA")

            if entry:
                grid_currents[label] = entry

        return grid_currents, applied, skipped

    def apply_phases_to_hardware(self, driver, grid_config, user=None, job_id=None):
        """Converts a phase grid to currents and writes it to the device."""
        grid_currents, applied, skipped = self.phases_to_currents(grid_config)

        channel_currents = {}
        for label, entry in grid_currents.items():
            theta_ch, phi_ch = self.label_map[label]
            if "theta" in entry:
                channel_currents[theta_ch] = entry["theta"]
            if "phi" in entry:
                channel_currents[phi_ch] = entry["phi"]

        apply_channel_currents(driver, channel_currents, user=user, job_id=job_id)

        if skipped:
            logging.info("skipped %d uncalibrated heater(s)", len(skipped))
        return grid_currents, applied, skipped

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def mesh_phases_to_grid(self, phases):
        """Turns the GUI/engine {mzi_id: {'theta','phi'}} dict into a grid."""
        return {label: copy.deepcopy(phases[label])
                for label in self.label_map if label in phases}

    def zero_config(self):
        """A grid that parks every mapped heater at zero phase."""
        return {label: {"theta": 0.0, "phi": 0.0} for label in self.label_map}
