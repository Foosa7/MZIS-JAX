import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit
from functools import partial
import itertools

# ──────────────────────────────────────────────────────────────────────────────
# Vectorized Ryser Permanent
# ──────────────────────────────────────────────────────────────────────────────

@jit
def ryser_permanent(M):
    """Computes the permanent of an n×n matrix using the vectorized Ryser formula.

    All 2^n subsets are enumerated simultaneously via bitmask operations,
    making this fully vectorized and differentiable through JAX.
    """
    n = M.shape[0]

    # Build bitmask matrix: (2^n - 1, n) boolean mask for each subset
    num_subsets = (1 << n) - 1                               # 2^n - 1
    subset_indices = jnp.arange(1, num_subsets + 1)          # [1, 2, ..., 2^n-1]
    bit_positions = jnp.arange(n)                            # [0, 1, ..., n-1]

    # (num_subsets, n) boolean: which columns are in each subset
    masks = ((subset_indices[:, None] >> bit_positions[None, :]) & 1).astype(M.dtype)

    # Row sums for each subset: (num_subsets, n_rows)
    # masks @ M^T gives sum of selected columns for each row
    row_sums = masks @ M.T  # (num_subsets, n)

    # Product of row sums for each subset
    products = jnp.prod(row_sums, axis=1)  # (num_subsets,)

    # Popcount for inclusion-exclusion signs
    popcounts = jnp.sum(masks.real.astype(jnp.int32), axis=1)
    is_odd = ((popcounts + n) % 2) == 1
    signs = jnp.where(is_odd, -1.0 + 0j, 1.0 + 0j)

    return jnp.sum(signs * products)


# ──────────────────────────────────────────────────────────────────────────────
# Differentiable Clements Mesh
# ──────────────────────────────────────────────────────────────────────────────

class Engine:
    def __init__(self, n_modes=8):
        """Initializes the photonic mesh with a default Clements layout and phases."""
        self.n_modes = n_modes
        self.layout = self._define_layout()

        # Build ordered list of MZI IDs and their mode indices
        self.mzi_ids = []
        self.mzi_modes = []  # list of (top_mode,) for each MZI
        for col in self.layout:
            for mzi in col:
                self.mzi_ids.append(mzi['id'])
                self.mzi_modes.append(mzi['mode_top'])

        self.n_mzis = len(self.mzi_ids)
        self._id_to_idx = {mid: i for i, mid in enumerate(self.mzi_ids)}

        # Precompute column boundaries: list of (start_idx, count) into the flat arrays
        self._col_slices = []
        idx = 0
        for col in self.layout:
            n = len(col)
            self._col_slices.append((idx, n))
            idx += n

        # Precompute mode pairs for each column as JAX arrays (for vectorized layer build)
        self._col_mode_tops = []
        for col in self.layout:
            tops = jnp.array([mzi['mode_top'] for mzi in col], dtype=jnp.int32)
            self._col_mode_tops.append(tops)

    def _define_layout(self):
        """Defines the MZI connectivity in a checkerboard (Clements) pattern."""
        cols = []
        def get_col_name(idx):
            """Converts a column index into a letter name."""
            chars = []
            while idx >= 0:
                chars.append(chr(65 + (idx % 26)))
                idx = idx // 26 - 1
            return "".join(reversed(chars))

        for i in range(self.n_modes):
            col_name = get_col_name(i)
            col_mzis = []
            # checkerboard pattern
            if i % 2 == 0:
                start_mode = 0
                count = self.n_modes // 2
            else:
                start_mode = 1
                count = (self.n_modes - 1) // 2

            for k in range(count):
                mzi_id = f"{col_name}{k+1}"
                col_mzis.append({'id': mzi_id, 'mode_top': start_mode + 2*k})
            cols.append(col_mzis)
        return cols

    # ──────────────────────────────────────────────────────────────────────
    # Layer and full unitary construction (JAX-native)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    @partial(jit, static_argnums=(3,))
    def _build_layer_matrix(thetas, phis, mode_tops, n_modes):
        """Builds a unitary matrix for one column of MZIs.

        Each MZI is an independent 2×2 block scattered into an n×n identity.
        Since MZIs in a column act on disjoint mode pairs they commute,
        so we compose them sequentially via jax.lax.scan.

        Args:
            thetas: (n_mzis_in_col,) internal phases
            phis: (n_mzis_in_col,) external phases
            mode_tops: (n_mzis_in_col,) top mode index for each MZI
            n_modes: total number of spatial modes (static)
        """
        U = jnp.eye(n_modes, dtype=jnp.complex128)
        n_mzis = thetas.shape[0]

        def scan_fn(U_acc, i):
            theta = thetas[i]
            phi = phis[i]
            top = mode_tops[i]

            t_half = theta / 2.0
            s = jnp.sin(t_half)
            c = jnp.cos(t_half)
            exp_p = jnp.exp(1j * phi)
            g = 1j * jnp.exp(1j * t_half)

            T = jnp.eye(n_modes, dtype=jnp.complex128)
            T = T.at[top, top].set(g * exp_p * s)
            T = T.at[top, top + 1].set(g * (-c))
            T = T.at[top + 1, top].set(g * exp_p * c)
            T = T.at[top + 1, top + 1].set(g * s)
            return U_acc @ T, None

        U_final, _ = jax.lax.scan(scan_fn, U, jnp.arange(n_mzis))
        return U_final

    def get_layer_matrix(self, thetas, phis, col_data):
        """Generates the unitary matrix for a single column of MZIs.

        Thin wrapper that slices into the phase arrays and calls the
        JIT-compiled static method.
        """
        # Find which column this is
        col_idx = None
        for i, col in enumerate(self.layout):
            if col is col_data:
                col_idx = i
                break

        if col_idx is None:
            # Fallback: match by MZI ids
            for i, col in enumerate(self.layout):
                if len(col) == len(col_data) and all(
                    a['id'] == b['id'] for a, b in zip(col, col_data)
                ):
                    col_idx = i
                    break

        start, count = self._col_slices[col_idx]
        col_thetas = thetas[start:start + count]
        col_phis = phis[start:start + count]
        mode_tops = self._col_mode_tops[col_idx]

        return self._build_layer_matrix(col_thetas, col_phis, mode_tops, self.n_modes)

    @staticmethod
    @partial(jit, static_argnums=(4,))
    def _compute_full_unitary(thetas, phis, col_slices_arr, col_mode_tops_padded, n_modes):
        """Computes the full mesh unitary from flat phase arrays.

        Args:
            thetas: (n_mzis,) all internal phases
            phis: (n_mzis,) all external phases
            col_slices_arr: (n_cols, 2) array of [start_idx, count] per column
            col_mode_tops_padded: (n_cols, max_mzis_per_col) padded mode tops
            n_modes: number of spatial modes (static)
        """
        n_cols = col_slices_arr.shape[0]

        def scan_body(U_acc, col_idx):
            start = col_slices_arr[col_idx, 0]
            count = col_slices_arr[col_idx, 1]
            max_count = col_mode_tops_padded.shape[1]

            # Dynamic slice into phase arrays
            col_thetas = jax.lax.dynamic_slice(thetas, (start,), (max_count,))
            col_phis = jax.lax.dynamic_slice(phis, (start,), (max_count,))
            col_tops = col_mode_tops_padded[col_idx]

            # Build layer — but we need to handle variable counts within
            # the fixed-shape scan. Use masking: only process `count` MZIs.
            U_layer = jnp.eye(n_modes, dtype=jnp.complex128)

            def mzi_body(U_l, j):
                theta = col_thetas[j]
                phi = col_phis[j]
                top = col_tops[j]

                t_half = theta / 2.0
                s = jnp.sin(t_half)
                c = jnp.cos(t_half)
                exp_p = jnp.exp(1j * phi)
                g = 1j * jnp.exp(1j * t_half)

                T = jnp.eye(n_modes, dtype=jnp.complex128)
                T = T.at[top, top].set(g * exp_p * s)
                T = T.at[top, top + 1].set(g * (-c))
                T = T.at[top + 1, top].set(g * exp_p * c)
                T = T.at[top + 1, top + 1].set(g * s)

                # Only apply if j < count (mask out padding)
                should_apply = j < count
                T_masked = jnp.where(should_apply, T, jnp.eye(n_modes, dtype=jnp.complex128))
                return U_l @ T_masked, None

            U_layer, _ = jax.lax.scan(mzi_body, U_layer, jnp.arange(max_count))
            return U_layer @ U_acc, None

        U0 = jnp.eye(n_modes, dtype=jnp.complex128)
        U_total, _ = jax.lax.scan(scan_body, U0, jnp.arange(n_cols))
        return U_total

    def compute_full_unitary(self, thetas, phis):
        """Computes the total unitary matrix of the entire Clements mesh."""
        # Prepare static arrays for the JIT-compiled function
        col_slices_arr = jnp.array(self._col_slices, dtype=jnp.int32)

        # Pad mode_tops to uniform shape for scan
        max_mzis = max(len(col) for col in self.layout)
        padded_tops = []
        for tops in self._col_mode_tops:
            pad_len = max_mzis - tops.shape[0]
            padded = jnp.pad(tops, (0, pad_len), constant_values=0)
            padded_tops.append(padded)
        col_mode_tops_padded = jnp.stack(padded_tops)

        pad_total = max_mzis
        thetas_padded = jnp.pad(thetas, (0, pad_total))
        phis_padded = jnp.pad(phis, (0, pad_total))

        return self._compute_full_unitary(
            thetas_padded, phis_padded, col_slices_arr,
            col_mode_tops_padded, self.n_modes
        )

    # ──────────────────────────────────────────────────────────────────────
    # Classical power flow
    # ──────────────────────────────────────────────────────────────────────

    def get_classical_flow(self, thetas, phis, input_powers, coherent=False):
        """Calculates power distribution across each layer.
        
        If coherent=True, models coherent wave interference (used for quantum mode visual proxy).
        If coherent=False, assumes incoherent light, so powers add linearly (classical power).
        """
        powers = []

        if coherent:
            state_c = jnp.array(jnp.sqrt(jnp.array(input_powers)), dtype=jnp.complex128)
            powers.append(jnp.abs(state_c) ** 2)

            for col_idx, col_data in enumerate(self.layout):
                start, count = self._col_slices[col_idx]
                col_thetas = thetas[start:start + count]
                col_phis = phis[start:start + count]
                mode_tops = self._col_mode_tops[col_idx]

                U_layer = self._build_layer_matrix(col_thetas, col_phis, mode_tops, self.n_modes)
                state_c = U_layer @ state_c
                powers.append(jnp.abs(state_c) ** 2)
        else:
            state_p = jnp.array(input_powers, dtype=jnp.float64)
            powers.append(state_p)

            for col_idx, col_data in enumerate(self.layout):
                start, count = self._col_slices[col_idx]
                col_thetas = thetas[start:start + count]
                col_phis = phis[start:start + count]
                mode_tops = self._col_mode_tops[col_idx]

                U_layer = self._build_layer_matrix(col_thetas, col_phis, mode_tops, self.n_modes)
                power_trans = jnp.abs(U_layer) ** 2
                state_p = power_trans @ state_p
                powers.append(state_p)

        return powers

    # ──────────────────────────────────────────────────────────────────────
    # Fock state propagation (vectorized via vmap)
    # ──────────────────────────────────────────────────────────────────────

    def propagate_fock(self, thetas, phis, input_occupation):
        """Simulates quantum photon propagation via vectorized permanents.

        Uses jax.lax.map to compute permanents for all output Fock states
        sequentially, keeping memory footprint low.
        """
        n_photons = sum(input_occupation)
        if n_photons == 0:
            return jnp.array([]), []

        U = self.compute_full_unitary(thetas, phis)

        # Input mode indices (repeated for boson occupation)
        in_indices = []
        for mode, count in enumerate(input_occupation):
            in_indices.extend([mode] * count)
        in_indices = jnp.array(in_indices, dtype=jnp.int32)

        # Enumerate all output Fock states
        basis = []
        for p in itertools.combinations_with_replacement(range(self.n_modes), n_photons):
            state = [0] * self.n_modes
            for m in p:
                state[m] += 1
            basis.append(tuple(state))

        # Build output index arrays for all basis states (padded to uniform length)
        out_indices_list = []
        for out_state in basis:
            out_idx = []
            for mode, count in enumerate(out_state):
                out_idx.extend([mode] * count)
            out_indices_list.append(out_idx)
        out_indices_arr = jnp.array(out_indices_list, dtype=jnp.int32)  # (n_basis, n_photons)

        # Input normalization factor
        in_factorials = jnp.array([float(_factorial(n)) for n in input_occupation])
        norm_in = jnp.prod(in_factorials)

        # Output normalization factors for all basis states
        out_factorials = jnp.array([
            [float(_factorial(n)) for n in state] for state in basis
        ])
        norm_out = jnp.prod(out_factorials, axis=1)  # (n_basis,)

        # Extract submatrices for all output states at once
        # U is (8, 8), out_indices_arr is (n_basis, n_photons)
        # We want to select rows out_indices_arr and columns in_indices for each basis state
        all_U_sub = U[out_indices_arr][:, :, in_indices]  # shape: (n_basis, n_photons, n_photons)

        # Map compute permanents using JAX lax.map to prevent OOM
        all_perms = jax.lax.map(ryser_permanent, all_U_sub)
        all_perm_sq = jnp.abs(all_perms) ** 2
        
        all_probs = all_perm_sq / (norm_in * norm_out)

        return all_probs, basis


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def _factorial(n):
    """Plain Python factorial for small integers (used in normalization constants)."""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result