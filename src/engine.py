import os
import math
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit
from functools import partial
import itertools
import json

# ──────────────────────────────────────────────────────────────────────────────
# Vectorized Ryser Permanent
# ──────────────────────────────────────────────────────────────────────────────

@jit
def ryser_permanent(M):
    """Computes the permanent of an nxn matrix using the vectorized Ryser formula.

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
# Vectorized Glynn Permanent (faster and less memory)
# ──────────────────────────────────────────────────────────────────────────────

@jit
def glynn_permanent(M):
    """Computes the permanent of an nxn matrix using the vectorized Glynn formula.
    
    Halves the memory and compute footprint compared to Ryser by fixing 
    the first polarization element to 1 and iterating over 2^(n-1) states.
    """
    n = M.shape[0]

    # 1. Generate 2^(n-1) bitmasks for the varying elements
    num_subsets = 1 << (n - 1)                                # 2^(n-1)
    indices = jnp.arange(num_subsets)                         # [0, 1, ..., 2^(n-1) - 1]
    bit_positions = jnp.arange(n - 1)                         # [0, 1, ..., n-2]
    
    # Binary matrix of shape (2^(n-1), n-1)
    binary_matrix = (indices[:, None] >> bit_positions[None, :]) & 1
    
    # 2. Build polarization vectors (+1 and -1)
    polarizations = jnp.where(binary_matrix == 1, -1.0, 1.0).astype(M.dtype)
    
    # Fix the first element to 1 to halve the computational space
    fixed_col = jnp.ones((num_subsets, 1), dtype=M.dtype)
    deltas = jnp.concatenate([fixed_col, polarizations], axis=1) # (2^(n-1), n)
    
    # 3. Vectorized matrix multiplication to get column sums
    # deltas @ M computes sum(delta_k * M_{k, j}) for all subsets simultaneously
    col_sums = deltas @ M  # (2^(n-1), n)
    
    # 4. Product of column sums for each polarization vector
    products = jnp.prod(col_sums, axis=1)  # (2^(n-1),)
    
    # 5. Compute signs for each term based on the number of -1s
    # The first column is always +1, so we only need to popcount the remaining n-1 columns
    popcounts = jnp.sum(binary_matrix, axis=1)
    signs = jnp.where(popcounts % 2 == 1, -1.0, 1.0).astype(M.dtype)
    
    # 6. Final sum and normalization by 2^(n-1)
    return jnp.sum(signs * products) / num_subsets


# ──────────────────────────────────────────────────────────────────────────────
# Differentiable Clements Mesh
# ──────────────────────────────────────────────────────────────────────────────

class Engine:
    def __init__(self, n_modes=8, bs_error=0.0):
        """Initializes the photonic mesh with a default Clements layout and phases.
        
        Args:
            n_modes: Number of spatial modes.
            bs_error: Beamsplitter error, modeled as deviation from 50:50 splitting.
                      Can be a scalar, or a tuple of (e_l, e_r) arrays for each MZI.
        """
        self.n_modes = n_modes
        self.bs_error = bs_error
        self.layout = self._define_layout()

        # Build ordered list of MZI IDs and their mode indices
        self.mzi_ids = []
        self.mzi_modes = []  # list of (mode_top,) for each MZI (the upper port index)
        for col in self.layout:
            for mzi in col:
                self.mzi_ids.append(mzi['id'])
                self.mzi_modes.append(mzi['mode_top'])

        self.n_mzis = len(self.mzi_ids)
        self._id_to_idx = {mid: i for i, mid in enumerate(self.mzi_ids)}
        
        # Initialize error arrays
        self.set_bs_error(self.bs_error)

        # Precompute column boundaries: list of (start_idx, count) into the flat arrays
        self._col_slices = []
        idx = 0
        for col in self.layout:
            n = len(col)
            self._col_slices.append((idx, n))
            idx += n

        # Precompute mode pairs(mode_top) for each column as JAX arrays (for vectorized layer build)
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



    # # If you pass a numpy.float32 or a jax.numpy.float32 scalar into this function,
    # # both of those isinstance checks will evaluate to False. 
    # # The code will fall through to the else block, treat the scalar as an array,
    # # and potentially crash later when JAX tries to broadcast shapes.

    # # The Fix: Use jnp.isscalar(). We use the fixed version

    def set_bs_error(self, bs_error):
        """Sets the global beamsplitter error and updates the internal arrays."""
        self.bs_error = bs_error
        
        # 1. Safely handle any scalar (Python int/float or JAX scalar)
        if isinstance(bs_error, (float, int)) or jnp.isscalar(bs_error):
            # Force float64 to match your jax_enable_x64 config
            self.e_l = jnp.full(self.n_mzis, float(bs_error), dtype=jnp.float64)
            self.e_r = jnp.full(self.n_mzis, float(bs_error), dtype=jnp.float64)
            
        elif isinstance(bs_error, tuple):
            self.e_l = jnp.asarray(bs_error[0], dtype=jnp.float64)
            self.e_r = jnp.asarray(bs_error[1], dtype=jnp.float64)
            
        else:
            self.e_l = jnp.asarray(bs_error, dtype=jnp.float64)
            self.e_r = self.e_l

    def load_calibration_errors(self, json_path, default_e=0.0):
        """
        Loads empirical beamsplitter errors from a node-isolation calibration JSON.
        Safely built using pure Python math to avoid JAX compilation crashes.
        """
        import math # Use standard math to prevent JAX from dispatching in the loop
        
        try:
            with open(json_path, 'r') as f:
                cal_data = json.load(f)
        except Exception as e:
            print(f"Failed to load calibration data: {e}")
            self.set_bs_error(default_e)
            return

        # 2. Flatten the dictionary lookup once
        phase_cal = cal_data.get('phase_calibration', {})
        e_l_list = []

        for mid in self.mzi_ids:
            key = f"{mid}_theta"
            e = default_e
            
            # 3. Safe, branchless dictionary extraction
            params = phase_cal.get(key, {}).get('phase_params', {})
            
            if params:
                A = params.get('amplitude', 0.0)
                C = params.get('offset', 1.0)
                
                if C + A > 0:
                    e_sq = (C - A) / (C + A)
                    # 4. standard math.sqrt instead of jnp.sqrt
                    e = math.sqrt(max(0.0, e_sq))
            
            e_l_list.append(e)
            
        # 5. Convert to JAX array exactly once at the end
        # e_l is the power splitting ratio error of the left BS, 
        # e_r is the power splitting ratio error of the right BS
        self.e_l = jnp.array(e_l_list, dtype=jnp.float64)
        self.e_r = self.e_l
        self.bs_error = (self.e_l, self.e_r) 
        

    # ──────────────────────────────────────────────────────────────────────
    # Layer and full unitary construction (JAX-native)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    @partial(jit, static_argnums=(5,))
    def _build_layer_matrix(thetas, phis, e_ls, e_rs, mode_tops, n_modes):
        """Builds a unitary matrix for one column of MZIs with support for beamsplitter errors.

        Args:
            thetas: (n_mzis_in_col,) internal phases
            phis: (n_mzis_in_col,) external phases
            e_ls: (n_mzis_in_col,) left beamsplitter error
            e_rs: (n_mzis_in_col,) right beamsplitter error
            mode_tops: (n_mzis_in_col,) top mode index for each MZI
            n_modes: total number of spatial modes (static)
        """
        U = jnp.eye(n_modes, dtype=jnp.complex128)
        n_mzis = thetas.shape[0]

        def scan_fn(U_acc, i):
            theta = thetas[i]
            phi = phis[i]
            top = mode_tops[i]
            el = e_ls[i]
            er = e_rs[i]

            sq_1_el = jnp.sqrt(1 + el)
            sq_1_ml = jnp.sqrt(1 - el)
            sq_1_er = jnp.sqrt(1 + er)
            sq_1_mr = jnp.sqrt(1 - er)

            bs_l_00 = sq_1_el / jnp.sqrt(2.0)
            bs_l_01 = 1j * sq_1_ml / jnp.sqrt(2.0)
            bs_l_10 = 1j * sq_1_ml / jnp.sqrt(2.0)
            bs_l_11 = sq_1_el / jnp.sqrt(2.0)

            bs_r_00 = sq_1_er / jnp.sqrt(2.0)
            bs_r_01 = 1j * sq_1_mr / jnp.sqrt(2.0)
            bs_r_10 = 1j * sq_1_mr / jnp.sqrt(2.0)
            bs_r_11 = sq_1_er / jnp.sqrt(2.0)

            p_0 = jnp.exp(1j * theta)
            
            mid_00 = p_0 * bs_l_00
            mid_01 = p_0 * bs_l_01
            mid_10 = bs_l_10
            mid_11 = bs_l_11

            out_00 = bs_r_00 * mid_00 + bs_r_01 * mid_10
            out_01 = bs_r_00 * mid_01 + bs_r_01 * mid_11
            out_10 = bs_r_10 * mid_00 + bs_r_11 * mid_10
            out_11 = bs_r_10 * mid_01 + bs_r_11 * mid_11
            
            # Apply original conventions: 
            # 1. negate column 1
            out_01 = -out_01
            out_11 = -out_11
            
            # 2. apply external phi to column 0
            exp_p = jnp.exp(1j * phi)
            out_00 = out_00 * exp_p
            out_10 = out_10 * exp_p

            T = jnp.eye(n_modes, dtype=jnp.complex128)
            T = T.at[top, top].set(out_00)
            T = T.at[top, top + 1].set(out_01)
            T = T.at[top + 1, top].set(out_10)
            T = T.at[top + 1, top + 1].set(out_11)
            
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
        col_els = self.e_l[start:start + count]
        col_ers = self.e_r[start:start + count]
        mode_tops = self._col_mode_tops[col_idx]

        return self._build_layer_matrix(col_thetas, col_phis, col_els, col_ers, mode_tops, self.n_modes)

    @staticmethod
    @partial(jit, static_argnums=(6,))
    def _compute_full_unitary(thetas, phis, e_ls, e_rs, col_slices_arr, col_mode_tops_padded, n_modes):
        """Computes the full mesh unitary from flat phase arrays.

        Args:
            thetas: (n_mzis,) all internal phases
            phis: (n_mzis,) all external phases
            e_ls: (n_mzis,) left errors
            e_rs: (n_mzis,) right errors
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
            col_els = jax.lax.dynamic_slice(e_ls, (start,), (max_count,))
            col_ers = jax.lax.dynamic_slice(e_rs, (start,), (max_count,))
            col_tops = col_mode_tops_padded[col_idx]

            # Build layer — but we need to handle variable counts within
            # the fixed-shape scan. Use masking: only process `count` MZIs.
            U_layer = jnp.eye(n_modes, dtype=jnp.complex128)

            def mzi_body(U_l, j):
                theta = col_thetas[j]
                phi = col_phis[j]
                top = col_tops[j]
                el = col_els[j]
                er = col_ers[j]

                sq_1_el = jnp.sqrt(1 + el)
                sq_1_ml = jnp.sqrt(1 - el)
                sq_1_er = jnp.sqrt(1 + er)
                sq_1_mr = jnp.sqrt(1 - er)

                bs_l_00 = sq_1_el / jnp.sqrt(2.0)
                bs_l_01 = 1j * sq_1_ml / jnp.sqrt(2.0)
                bs_l_10 = 1j * sq_1_ml / jnp.sqrt(2.0)
                bs_l_11 = sq_1_el / jnp.sqrt(2.0)

                bs_r_00 = sq_1_er / jnp.sqrt(2.0)
                bs_r_01 = 1j * sq_1_mr / jnp.sqrt(2.0)
                bs_r_10 = 1j * sq_1_mr / jnp.sqrt(2.0)
                bs_r_11 = sq_1_er / jnp.sqrt(2.0)

                p_0 = jnp.exp(1j * theta)
                
                mid_00 = p_0 * bs_l_00
                mid_01 = p_0 * bs_l_01
                mid_10 = bs_l_10
                mid_11 = bs_l_11

                out_00 = bs_r_00 * mid_00 + bs_r_01 * mid_10
                out_01 = bs_r_00 * mid_01 + bs_r_01 * mid_11
                out_10 = bs_r_10 * mid_00 + bs_r_11 * mid_10
                out_11 = bs_r_10 * mid_01 + bs_r_11 * mid_11
                
                out_01 = -out_01
                out_11 = -out_11
                
                exp_p = jnp.exp(1j * phi)
                out_00 = out_00 * exp_p
                out_10 = out_10 * exp_p

                T = jnp.eye(n_modes, dtype=jnp.complex128)
                T = T.at[top, top].set(out_00)
                T = T.at[top, top + 1].set(out_01)
                T = T.at[top + 1, top].set(out_10)
                T = T.at[top + 1, top + 1].set(out_11)

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
        els_padded = jnp.pad(self.e_l, (0, pad_total))
        ers_padded = jnp.pad(self.e_r, (0, pad_total))

        # 1. Get the mathematically perfect internal matrix from the GPU
        U_internal = self._compute_full_unitary(
            thetas_padded, phis_padded, els_padded, ers_padded, col_slices_arr,
            col_mode_tops_padded, self.n_modes
        )

        # ──────────────────────────────────────────────────────────────────
        # 2. FACET COUPLING WRAPPER (The "Reality" Layer)
        # ──────────────────────────────────────────────────────────────────
        if self.use_calibration:
            p_in = 3.0
            p_out_measured = jnp.array([
                0.062, 0.062, 0.049, 0.050, 0.057, 0.066, 0.038, 0.047
            ], dtype=jnp.float64)

            # Assuming symmetric loss across input and output facets: C_in = C_out = sqrt(T)
            transmission = p_out_measured / p_in
            coupling_coeffs = jnp.sqrt(transmission)
            C = jnp.diag(coupling_coeffs)

            # Wrap the perfect internal matrix with physical coupling
            U_real = C @ U_internal @ C
            return U_real
            
        else:
            # If no calibration is loaded, return the perfect math matrix
            return U_internal

    # def compute_full_unitary(self, thetas, phis):
    #     """Computes the total unitary matrix of the entire Clements mesh."""
    #     # Prepare static arrays for the JIT-compiled function
    #     col_slices_arr = jnp.array(self._col_slices, dtype=jnp.int32)

    #     # Pad mode_tops to uniform shape for scan
    #     max_mzis = max(len(col) for col in self.layout)
    #     padded_tops = []
    #     for tops in self._col_mode_tops:
    #         pad_len = max_mzis - tops.shape[0]
    #         padded = jnp.pad(tops, (0, pad_len), constant_values=0)
    #         padded_tops.append(padded)
    #     col_mode_tops_padded = jnp.stack(padded_tops)

    #     pad_total = max_mzis
    #     thetas_padded = jnp.pad(thetas, (0, pad_total))
    #     phis_padded = jnp.pad(phis, (0, pad_total))
    #     els_padded = jnp.pad(self.e_l, (0, pad_total))
    #     ers_padded = jnp.pad(self.e_r, (0, pad_total))

    #     return self._compute_full_unitary(
    #         thetas_padded, phis_padded, els_padded, ers_padded, col_slices_arr,
    #         col_mode_tops_padded, self.n_modes
    #     )

    # ──────────────────────────────────────────────────────────────────────
    # Classical power flow
    # ──────────────────────────────────────────────────────────────────────

    def get_classical_flow(self, thetas, phis, input_powers, coherent=True):
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
                col_els = self.e_l[start:start + count]
                col_ers = self.e_r[start:start + count]
                mode_tops = self._col_mode_tops[col_idx]

                U_layer = self._build_layer_matrix(col_thetas, col_phis, col_els, col_ers, mode_tops, self.n_modes)
                state_c = U_layer @ state_c
                powers.append(jnp.abs(state_c) ** 2)
        else:
            state_p = jnp.array(input_powers, dtype=jnp.float64)
            powers.append(state_p)

            for col_idx, col_data in enumerate(self.layout):
                start, count = self._col_slices[col_idx]
                col_thetas = thetas[start:start + count]
                col_phis = phis[start:start + count]
                col_els = self.e_l[start:start + count]
                col_ers = self.e_r[start:start + count]
                mode_tops = self._col_mode_tops[col_idx]

                U_layer = self._build_layer_matrix(col_thetas, col_phis, col_els, col_ers, mode_tops, self.n_modes)
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
        all_perms = jax.lax.map(glynn_permanent, all_U_sub)
        all_perm_sq = jnp.abs(all_perms) ** 2
        
        all_probs = all_perm_sq / (norm_in * norm_out)

        return all_probs, basis


    # ──────────────────────────────────────────────────────────────────────
    # Hardware Calibration & Phase Constraints (Neurophox Integration)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    @jit
    def apply_phase_constraints(thetas, phis):
        """
        Enforces realistic hardware constraints on the phases, ensuring they 
        reside within the bounds [0, 2π). Similar to neurophox's common mode flow, 
        this ensures DACs/heaters don't receive negative or out-of-bound voltage requests.
        """
        # Map phases into standard [0, 2π) bounds
        thetas_norm = jnp.mod(thetas, 2 * jnp.pi)
        phis_norm = jnp.mod(phis, 2 * jnp.pi)
        
        # If theta is in [π, 2π), we can reflect it to [0, π) by flipping phi
        # since an MZI with internal phase θ + π is equivalent to θ with a π shift on phi.
        # This reduces maximum required heating power!
        reflect_mask = thetas_norm > jnp.pi
        
        thetas_opt = jnp.where(reflect_mask, 2 * jnp.pi - thetas_norm, thetas_norm)
        phis_opt = jnp.where(reflect_mask, jnp.mod(phis_norm + jnp.pi, 2 * jnp.pi), phis_norm)
        
        return thetas_opt, phis_opt

    @staticmethod
    @jit
    def parallel_nullification(amps_in, e_l, e_r):
        """
        JAX-native implementation of parallel nullification for a single layer of MZIs.
        Given input coherent optical amplitudes (amps_in), calculates the required 
        internal (θ) and external (φ) phases to perfectly route all power to the bar ports.
        
        This mimics the experimental self-calibration routine.
        
        Args:
            amps_in: Complex input amplitudes to the MZI (shape: [2])
            e_l, e_r: Beamsplitter errors
        Returns:
            theta, phi required for nullification
        """
        upper_in = amps_in[0]
        lower_in = amps_in[1]
        
        # Calculate ideal required phases
        # phi compensates for the relative phase difference between inputs
        phi_req = jnp.angle(upper_in / (lower_in + 1e-12))
        
        # theta dictates the splitting ratio to counteract the power imbalance
        # We need to consider the beamsplitter errors here for perfect nullification
        sq_1_el = jnp.sqrt(1 + e_l)
        sq_1_ml = jnp.sqrt(1 - e_l)
        
        # Power ratio considering the left defective beamsplitter
        power_ratio = jnp.abs(upper_in / (lower_in + 1e-12))
        theta_req = jnp.arctan(power_ratio * (sq_1_ml / sq_1_el)) * 2.0
        
        return jnp.mod(theta_req, 2 * jnp.pi), jnp.mod(phi_req, 2 * jnp.pi)


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