"""Thermal crosstalk kernels on the physical mesh layout.

Two kernels, both producing a `(n_heaters, n_heaters)` coupling matrix `C` such
that the phase perturbation on heater `v`'s segment from the dissipated powers
`P` (mW) of all heaters is `dphase[v] = sum_s C[s, v] * P[s]`.

`grid_isotropic` is the previous model -- `amp * exp(-d_grid / lam)` on
`(column, mode_top)` index distance -- kept as an ablation control.

`image_sink` is the physical one. A heater is a finite line source on the chip
surface; the substrate sits on a TEC, so the boundary is approximately an
isothermal plane at depth `h`, whose image gives

    g(rho) = 1/sqrt(rho^2 + a^2) - 1/sqrt(rho^2 + a^2 + 4 h^2)

for the surface temperature a distance `rho` from a surface point source. Near
field this is ~1/rho; beyond the sink depth it crosses over to ~2h^2/rho^3.
Both the source line and the victim waveguide segment are integrated by Gauss
-Legendre quadrature, so nothing here assumes a far-field point approximation
-- which matters, since the 190 um heater and its 254 um neighbour are the same
order of magnitude apart.

The kernel is evaluated *differentially*: `g` on the victim's own arm minus `g`
on its partner arm, one mode pitch away. Common-mode heating does not shift an
MZI's phase, so this is what actually reaches the model. It also encodes the
selection rule -- near heaters straddle the 127 um arm pair asymmetrically and
survive the subtraction; far heaters are nearly equidistant from both arms and
cancel.

All parameters are fitted in log space where they must stay positive.
"""

import jax
import jax.numpy as jnp
import numpy as np

from src.geometry import MeshGeometry


class ThermalKernel:
    """Precomputes the fixed geometry, leaves the physics parameters free."""

    # Peak pairwise coupling corresponding to amp = 1, in rad/mW. Chosen so a
    # unit amplitude is a large-but-not-absurd effect: at ~12 mW per heater it
    # is ~0.12 rad of phase error on the worst-coupled neighbour.
    AMP_UNIT = 1e-2

    def __init__(self, geometry: MeshGeometry, n_quad=8):
        self.geom = geometry
        d_own, d_partner, w = geometry.arm_distances(n_quad=n_quad)
        self.d_own = jnp.asarray(d_own)
        self.d_partner = jnp.asarray(d_partner)
        self.w = jnp.asarray(w)
        self.mask = jnp.asarray(geometry.self_mask())
        self.d_grid = jnp.asarray(geometry.grid_distances())
        self.n_heaters = geometry.n_heaters

    # ──────────────────────────────────────────────────────────────────────

    def init_params(self, mode):
        """Sensible starting point for each kernel mode.

        `h` starts near a typical thinned-die thickness and `a` near the heater
        half-width; both are free, but starting them at the physical value keeps
        the optimiser out of the degenerate large-h regime where the image term
        vanishes and the kernel collapses to a bare 1/rho.
        """
        if mode == "none":
            return {}
        if mode == "grid_isotropic":
            return {
                "amp": jnp.asarray(1e-3),
                "log_lam": jnp.asarray(np.log(1.0)),
            }
        if mode == "image_sink":
            return {
                "amp": jnp.asarray(1e-2),
                "log_h": jnp.asarray(np.log(400.0)),   # um, substrate sink depth
                "log_a": jnp.asarray(np.log(15.0)),    # um, source core radius
            }
        raise ValueError("unknown kernel mode %r" % mode)

    def coupling(self, params, mode):
        """`(n_heaters, n_heaters)` matrix, mW -> rad.

        The geometric part is normalised to unit peak, so `amp` means "peak
        pairwise coupling, in units of AMP_UNIT rad/mW" regardless of the shape
        parameters. Two reasons. It keeps `amp` O(1), which matters because
        Adam's step is set in the parameter's own units and a raw 1e-3-scale
        amplitude under lr=0.05 would only oscillate. And it keeps `amp` linear
        rather than logged, so it can sit at exactly zero with a live gradient
        and move to either sign -- a log-amplitude has a gradient proportional
        to the amplitude itself, which pins a zero start in place and manufactures
        a false null.
        """
        if mode == "none":
            return jnp.zeros((self.n_heaters, self.n_heaters))

        if mode == "grid_isotropic":
            lam = jnp.exp(params["log_lam"])
            c = jnp.exp(-self.d_grid / lam) * self.mask
        else:
            h = jnp.exp(params["log_h"])
            a2 = jnp.exp(params["log_a"]) ** 2
            four_h2 = 4.0 * h * h

            def g(d):
                r2 = d * d + a2
                return jax.lax.rsqrt(r2) - jax.lax.rsqrt(r2 + four_h2)

            diff = g(self.d_own) - g(self.d_partner)
            # Average over the source line and victim segment.
            c = jnp.einsum("svij,ij->sv", diff, self.w) * self.mask

        peak = jnp.max(jnp.abs(c))
        return params["amp"] * self.AMP_UNIT * c / jnp.where(peak > 0, peak, 1.0)

    # ──────────────────────────────────────────────────────────────────────

    def report(self, params, mode):
        """Human-readable coupling strengths at the characteristic separations."""
        c = np.asarray(self.coupling(params, mode))
        geom = self.geom
        xy = geom.heater_xy
        d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
        out = {}
        for label, target in (
            ("same_column_254um", 2 * geom.mode_pitch),
            ("intra_mzi_%dum" % round(geom.intra_mzi_x), geom.intra_mzi_x),
            ("next_column_%dum" % round(geom.column_pitch), geom.column_pitch),
        ):
            sel = np.abs(d - target) < 1.0
            sel &= ~np.eye(geom.n_heaters, dtype=bool)
            if sel.any():
                out[label] = float(np.abs(c[sel]).mean())
        out["max_abs"] = float(np.abs(c).max())
        return out


def total_power_term(params, powers):
    """Global mesh-loading term: every segment responds to total dissipated power.

    A pairwise kernel of any shape cannot represent a gradient across a mesh
    that spans millimetres, because the heat path to the TEC differs end to end.
    This adds that missing rank-one direction: one coefficient per heater
    against the summed power of the whole mesh.
    """
    if "global_beta" not in params:
        return 0.0
    return params["global_beta"] * powers.sum(axis=-1, keepdims=True)
