"""Physical layout of the mesh: where every heater actually sits on the die.

`layout/<n>mode_geometry.json` is generated from the tapeout GDS by
`scripts/extract_layout_geometry.py`. This module turns it into arrays indexed
the way the rest of the codebase expects, and precomputes the source/victim
distance tensors a thermal kernel needs.

Ordering. `heater_xy` is interleaved `[theta, phi]` per MZI in `Engine.mzi_ids`
order, which is exactly the 56-vector ordering of `currents_mA` in the
DigitalTwin datasets and of `build_heater_names` in their generator. Index
`2*i` is `mzi_ids[i]_theta`, `2*i+1` is `mzi_ids[i]_phi`.

Why the geometry matters. A heater is a ~190 um line source; its nearest
neighbour is 254 um away, and the next Clements column is 1125 um away. In
*grid* coordinates (column index, mode index) both of those are "distance 1",
so an isotropic `exp(-d_grid/lambda)` kernel is forced to average a strong
coupling against a weak one. Working in microns removes that degeneracy.

The other half is that phase error is *differential*. A heater's power shifts
both arms of a neighbouring MZI; only the difference survives into that MZI's
phase. Near heaters sit asymmetrically across the 127 um arm pair and couple
differentially; distant heaters are nearly equidistant from both arms and
largely cancel. So the tensors below are always built per-arm.
"""

import json
import os

import numpy as np

_LAYOUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "layout")


class MeshGeometry:
    """Physical heater coordinates for an n-mode Clements mesh."""

    def __init__(self, data):
        self.raw = data
        self.n_modes = data["n_modes"]
        self.mode_pitch = data["mode_pitch"]
        self.heater_length = data["heater_length"]
        self.segment_length = data["segment_length"]
        self.column_pitch = data["column_pitch_x"]
        self.intra_mzi_x = data["intra_mzi_x"]

        # Engine order: column-major, MZI 1..k within each column.
        mzis = data["mzis"]
        self.mzi_ids = sorted(mzis, key=lambda k: (mzis[k]["column"], mzis[k]["mode_top"]))
        self.n_mzis = len(self.mzi_ids)

        self.column = np.array([mzis[m]["column"] for m in self.mzi_ids], dtype=np.int32)
        self.mode_top = np.array([mzis[m]["mode_top"] for m in self.mzi_ids], dtype=np.int32)

        theta_xy = np.array([mzis[m]["theta_xy"] for m in self.mzi_ids], dtype=np.float64)
        phi_xy = np.array([mzis[m]["phi_xy"] for m in self.mzi_ids], dtype=np.float64)

        # Interleave to match the dataset current vector.
        self.heater_xy = np.empty((2 * self.n_mzis, 2), dtype=np.float64)
        self.heater_xy[0::2] = theta_xy
        self.heater_xy[1::2] = phi_xy

        self.heater_names = []
        for m in self.mzi_ids:
            self.heater_names.append("%s_theta" % m)
            self.heater_names.append("%s_phi" % m)

        self.n_heaters = len(self.heater_names)
        self.heater_mzi = np.repeat(np.arange(self.n_mzis), 2).astype(np.int32)
        self.heater_is_theta = np.tile(np.array([True, False]), self.n_mzis)

        # Both heaters of an MZI sit on the mode_top arm; the partner arm is one
        # mode pitch away, toward increasing mode index.
        sign = -1.0 if data["mode0_at"] == "high" else 1.0
        self.partner_xy = self.heater_xy.copy()
        self.partner_xy[:, 1] += sign * self.mode_pitch

    @classmethod
    def load(cls, n_modes, path=None):
        path = path or os.path.join(_LAYOUT_DIR, "%dmode_geometry.json" % n_modes)
        if not os.path.exists(path):
            raise FileNotFoundError(
                "%s not found -- generate it with scripts/extract_layout_geometry.py" % path
            )
        with open(path) as fh:
            return cls(json.load(fh))

    def index(self, name):
        """Index of a heater by `<mzi_id>_<theta|phi>` name."""
        return self.heater_names.index(name)

    # ──────────────────────────────────────────────────────────────────────
    # Distance tensors
    # ──────────────────────────────────────────────────────────────────────

    def arm_distances(self, n_quad=8):
        """Source-to-victim-arm distances, in microns.

        Each source heater is a line of `heater_length` along propagation. Each
        victim heater defines the waveguide stretch its phase is lumped onto: a
        segment of `segment_length` centred on that heater, on the mode_top arm,
        with a partner segment one mode pitch away.

        Returns `(d_own, d_partner, w)` where the distance arrays are
        `(n_heaters, n_heaters, n_quad, n_quad)` -- indexed
        `[source, victim, source_node, victim_node]` -- and `w` is the
        `(n_quad, n_quad)` product quadrature weight, normalised to sum to 1 so
        the result is a *mean* over the source line and victim segment rather
        than a length-scaled integral.
        """
        nodes, weights = np.polynomial.legendre.leggauss(n_quad)
        w = np.outer(weights, weights) / 4.0  # both intervals are half-width scaled

        src_x = self.heater_xy[:, 0][:, None] + 0.5 * self.heater_length * nodes[None, :]
        src_y = self.heater_xy[:, 1]
        vic_x = self.heater_xy[:, 0][:, None] + 0.5 * self.segment_length * nodes[None, :]

        dx = src_x[:, None, :, None] - vic_x[None, :, None, :]

        dy_own = (src_y[:, None] - self.heater_xy[None, :, 1])[:, :, None, None]
        dy_partner = (src_y[:, None] - self.partner_xy[None, :, 1])[:, :, None, None]

        d_own = np.sqrt(dx * dx + dy_own * dy_own)
        d_partner = np.sqrt(dx * dx + dy_partner * dy_partner)
        return d_own, d_partner, w

    def self_mask(self):
        """`(n_heaters, n_heaters)` mask that zeroes each heater's own entry.

        A heater's effect on its own segment is the intended actuation, already
        carried by the per-heater phase scale and gain, so it must not be
        double-counted as crosstalk. Everything off-diagonal is real coupling --
        including a theta heater onto its own MZI's phi segment.
        """
        m = np.ones((self.n_heaters, self.n_heaters), dtype=np.float64)
        np.fill_diagonal(m, 0.0)
        return m

    def grid_distances(self):
        """The old `(column, mode_top)` grid distance, for the ablation baseline.

        This is what the previous isotropic fit used. Kept so the physical
        kernel can be compared against it on identical footing.
        """
        col = self.column[self.heater_mzi].astype(np.float64)
        row = self.mode_top[self.heater_mzi].astype(np.float64)
        dc = col[:, None] - col[None, :]
        dr = row[:, None] - row[None, :]
        return np.sqrt(dc * dc + dr * dr)

    def summary(self):
        d_own, d_partner, w = self.arm_distances(n_quad=8)
        mean_own = (d_own * w).sum(axis=(2, 3))
        off = mean_own[~np.eye(self.n_heaters, dtype=bool)]
        return {
            "n_modes": self.n_modes,
            "n_mzis": self.n_mzis,
            "n_heaters": self.n_heaters,
            "mode_pitch_um": self.mode_pitch,
            "heater_length_um": self.heater_length,
            "segment_length_um": self.segment_length,
            "column_pitch_um": self.column_pitch,
            "nearest_heater_um": float(
                np.min(
                    np.linalg.norm(
                        self.heater_xy[:, None, :] - self.heater_xy[None, :, :], axis=-1
                    )
                    + np.eye(self.n_heaters) * 1e9
                )
            ),
            "min_mean_arm_distance_um": float(off.min()),
        }


if __name__ == "__main__":
    for n in (8, 12):
        try:
            g = MeshGeometry.load(n)
        except FileNotFoundError as exc:
            print(exc)
            continue
        print("--- %d modes" % n)
        for k, v in g.summary().items():
            print("  %-26s %s" % (k, v))
