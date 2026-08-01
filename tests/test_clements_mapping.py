import numpy as np
import jax.numpy as jnp
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.engine import Engine
from decompose.pnn import decompose_clements, clements_to_engine_phases


def _haar_unitary(dim, rng):
    Z = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(Z)
    return Q @ np.diag(np.diagonal(R) / np.abs(np.diagonal(R)))


def _program(engine, U_target):
    """Decomposes U_target and returns what the mesh actually produces."""
    phis, thetas, alphas = decompose_clements(U_target.copy(), block='mzi')
    settings, output_phases = clements_to_engine_phases(phis, thetas, alphas, engine.layout)

    thetas_arr = jnp.array([settings[mid][0] for mid in engine.mzi_ids])
    phis_arr = jnp.array([settings[mid][1] for mid in engine.mzi_ids])
    U = np.asarray(engine.compute_full_unitary(thetas_arr, phis_arr))
    return U, output_phases


def test_decomposition_round_trip():
    print("=== Clements decomposition -> Engine phases round-trip ===")
    rng = np.random.default_rng(0)
    worst = 0.0

    for n_modes in (4, 5, 6, 8, 12):
        engine = Engine(n_modes=n_modes, bs_error=0.0)
        U_target = _haar_unitary(n_modes, rng)
        U, output_phases = _program(engine, U_target)

        # The mesh has no output phase shifters, so it reproduces the target only
        # up to the residual screen. Power routing (|U|) must match exactly.
        amp_err = np.abs(np.abs(U) - np.abs(U_target)).max()
        full_err = np.abs(U - np.diag(np.exp(1j * output_phases)) @ U_target).max()
        worst = max(worst, full_err)

        print(f"  N={n_modes:2d}  max||U|-|U_target|| = {amp_err:.2e}   "
              f"max|U - D@U_target| = {full_err:.2e}")
        assert full_err < 1e-9, f"round-trip failed for N={n_modes}: {full_err}"

    print(f"  worst case: {worst:.2e}\n")


def test_permutation_routing():
    print("=== Switching (permutation) matrices ===")
    rng = np.random.default_rng(7)

    for n_modes in (8, 12):
        engine = Engine(n_modes=n_modes, bs_error=0.0)
        P_target = np.eye(n_modes)[rng.permutation(n_modes)].astype(np.complex128)
        U, _ = _program(engine, P_target)

        err = np.abs(np.abs(P_target) ** 2 - np.abs(U) ** 2).max()
        print(f"  N={n_modes:2d}  max|P_target - |U|^2| = {err:.2e}")
        assert err < 1e-9, f"switching failed for N={n_modes}: {err}"
    print()


def test_naive_mapping_is_wrong():
    """Guards the fix: theta_engine = 2*theta_pnn silently corrupts the target."""
    print("=== Naive 2*theta mapping (the bug this replaced) ===")
    engine = Engine(n_modes=8, bs_error=0.0)
    U_target = _haar_unitary(8, np.random.default_rng(0))

    phis, thetas, _ = decompose_clements(U_target.copy(), block='mzi')
    th = np.zeros(engine.n_mzis)
    ph = np.zeros(engine.n_mzis)
    for col_idx, col in enumerate(engine.layout):
        p = col_idx // 2
        for mzi in col:
            q = mzi['mode_top']
            i = engine._id_to_idx[mzi['id']]
            th[i] = np.mod(2 * thetas[q, p], 2 * np.pi)
            ph[i] = np.mod(phis[q, p], 2 * np.pi)

    U_naive = np.asarray(engine.compute_full_unitary(jnp.array(th), jnp.array(ph)))
    naive_err = np.abs(np.abs(U_target) ** 2 - np.abs(U_naive) ** 2).max()
    U_fixed, _ = _program(engine, U_target)
    fixed_err = np.abs(np.abs(U_target) ** 2 - np.abs(U_fixed) ** 2).max()

    print(f"  naive mapping: max transfer-matrix error = {naive_err:.4f}")
    print(f"  fixed mapping: max transfer-matrix error = {fixed_err:.2e}")
    assert naive_err > 0.1, "naive mapping unexpectedly accurate -- has a convention changed?"
    assert fixed_err < 1e-9
    print()


if __name__ == "__main__":
    test_decomposition_round_trip()
    test_permutation_routing()
    test_naive_mapping_is_wrong()
    print("All Clements mapping tests passed.")
