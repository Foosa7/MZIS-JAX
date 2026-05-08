import numpy as np
import itertools
import jax
import jax.numpy as jnp
import optax
from scipy.stats import unitary_group
from scipy.optimize import minimize
from src.engine import glynn_permanent, _factorial

# Detect GPU availability and set performance tiers
_HAS_GPU = jax.default_backend() == 'gpu'

# GPU: push hard — plenty of VRAM and parallelism
# CPU: conservative — avoid hogging the main thread for too long
if _HAS_GPU:
    _CLASSICAL_RESTARTS = 500
    _CLASSICAL_ITERS = 300
    _QUANTUM_RESTARTS = 200
    _QUANTUM_ITERS = 250
    _LR = 0.03
else:
    _CLASSICAL_RESTARTS = 50
    _CLASSICAL_ITERS = 150
    _QUANTUM_RESTARTS = 30
    _QUANTUM_ITERS = 100
    _LR = 0.05

print(f"[StateRouter] Backend: {jax.default_backend()} | "
      f"Classical: {_CLASSICAL_RESTARTS}x{_CLASSICAL_ITERS} | "
      f"Quantum: {_QUANTUM_RESTARTS}x{_QUANTUM_ITERS} | LR: {_LR}")

class StateRouter:
    @staticmethod
    def get_unitary_mapping_state_to_basis(psi):
        """
        Returns a unitary V such that V * |0> = psi
        psi is an N-dimensional complex vector (normalized).
        """
        N = len(psi)
        psi = np.asarray(psi, dtype=np.complex128)
        
        # QR decomposition to complete the basis
        mat = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        mat[:, 0] = psi
        Q, R = np.linalg.qr(mat)
        
        # Adjust phase of the first column so Q[:, 0] exactly matches psi
        phase = R[0, 0] / np.abs(R[0, 0])
        Q[:, 0] = Q[:, 0] * phase
        
        return Q

    @staticmethod
    def generate_routing_unitaries(psi_in, psi_out, num_unitaries=10):
        """
        Generates `num_unitaries` different unitary matrices U such that
        U @ psi_in = psi_out (up to a global phase or exactly).
        """
        N = len(psi_in)
        V_in = StateRouter.get_unitary_mapping_state_to_basis(psi_in)
        V_out = StateRouter.get_unitary_mapping_state_to_basis(psi_out)
        
        unitaries = []
        for _ in range(num_unitaries):
            if N > 1:
                U_sub = unitary_group.rvs(N - 1)
            else:
                U_sub = np.eye(0)
                
            W_inner = np.eye(N, dtype=np.complex128)
            if N > 1:
                W_inner[1:, 1:] = U_sub
                
            U = V_out @ W_inner @ V_in.conj().T
            unitaries.append(U)
            
            
        return unitaries

    @staticmethod
    def _run_vmap_optimization(engine, loss_fn, num_restarts=1000, max_iters=200):
        """
        Core vmap optimization engine. Takes a loss_fn(phases) and runs
        num_restarts parallel Adam optimizations, returning deduplicated results
        sorted by loss.
        """
        n_mzis = len(engine.mzi_ids)

        optimizer = optax.adam(learning_rate=_LR)

        def single_optimization(init_phases):
            opt_state = optimizer.init(init_phases)

            def step(carry, _):
                params, state = carry
                loss_val, grads = jax.value_and_grad(loss_fn)(params)
                updates, state = optimizer.update(grads, state)
                params = optax.apply_updates(params, updates)
                return (params, state), loss_val

            (final_params, _), loss_history = jax.lax.scan(
                step, (init_phases, opt_state), None, length=max_iters
            )
            return final_params, loss_history[-1]

        parallel_optimization = jax.jit(jax.vmap(single_optimization))

        key = jax.random.PRNGKey(42)
        init_phases_batch = jax.random.uniform(
            key, shape=(num_restarts, 2 * n_mzis), minval=0, maxval=2*jnp.pi
        )

        final_phases_batch, final_losses = parallel_optimization(init_phases_batch)

        # --- Deduplication and filtering ---
        success_threshold = 1e-4
        successful_mask = final_losses < success_threshold
        
        valid_phases = np.array(final_phases_batch[successful_mask])
        valid_losses = np.array(final_losses[successful_mask])

        unique_options = []
        unique_losses = []
        tolerance = 1e-2

        for p, l in zip(valid_phases, valid_losses):
            p_wrapped = np.mod(p, 2 * np.pi)
            is_new = True
            for unique_p in unique_options:
                diff = np.abs(p_wrapped - unique_p)
                diff = np.minimum(diff, 2*np.pi - diff)
                if np.mean(diff) < tolerance:
                    is_new = False
                    break
                    
            if is_new:
                unique_options.append(p_wrapped)
                unique_losses.append(l)
                
        # Fallback: if nothing passed the threshold, return the single best run
        if not unique_options:
            best_idx = jnp.argmin(final_losses)
            best_phases = np.array(final_phases_batch[best_idx])
            best_loss = float(final_losses[best_idx])
            unique_options.append(np.mod(best_phases, 2*np.pi))
            unique_losses.append(best_loss)

        results = []
        for p, l in zip(unique_options, unique_losses):
            thetas = p[:n_mzis]
            phis = p[n_mzis:]
            results.append((thetas, phis, float(l)))
            
        results.sort(key=lambda x: x[2])
        return results

    @staticmethod
    def optimize_coherent_routing_vmap(engine, psi_in, psi_target, 
                                        num_restarts=_CLASSICAL_RESTARTS, 
                                        max_iters=_CLASSICAL_ITERS):
        """
        Optimizes MZI phases so that |U @ psi_in|^2 matches |psi_target|^2.
        Works for a single coherent input state. Uses field-level simulation.
        """
        psi_in = jnp.array(psi_in, dtype=jnp.complex128)
        P_target = jnp.abs(jnp.array(psi_target, dtype=jnp.complex128))**2
        
        # Normalize target power to match input power
        target_sum = jnp.sum(P_target)
        input_power = jnp.sum(jnp.abs(psi_in)**2)
        if target_sum > 0:
            P_target = P_target * (input_power / target_sum)
        
        n_mzis = len(engine.mzi_ids)

        def loss_fn(phases):
            thetas = jnp.mod(phases[:n_mzis], 2 * jnp.pi)
            phis = jnp.mod(phases[n_mzis:], 2 * jnp.pi)
            
            U = engine.compute_full_unitary(thetas, phis)
            psi_out = U @ psi_in
            P_out = jnp.abs(psi_out)**2
            return jnp.mean((P_out - P_target)**2)

        return StateRouter._run_vmap_optimization(engine, loss_fn, num_restarts, max_iters)

    @staticmethod
    def optimize_incoherent_routing_vmap(engine, P_in, P_target, 
                                          num_restarts=_CLASSICAL_RESTARTS, 
                                          max_iters=_CLASSICAL_ITERS):
        """
        Pure JAX implementation. Runs `num_restarts` optimizations in parallel
        on the GPU, without strict boundary walls.
        Returns a list of unique phase configurations that successfully routed the light.
        """
        P_in = jnp.array(P_in, dtype=jnp.float64)
        P_target = jnp.array(P_target, dtype=jnp.float64)
        
        target_sum = jnp.sum(P_target)
        if target_sum > 0:
            P_target = P_target * (jnp.sum(P_in) / target_sum)

        n_mzis = len(engine.mzi_ids)

        def loss_fn(phases):
            thetas = jnp.mod(phases[:n_mzis], 2 * jnp.pi)
            phis = jnp.mod(phases[n_mzis:], 2 * jnp.pi)
            
            U = engine.compute_full_unitary(thetas, phis)
            power_trans = jnp.abs(U)**2
            P_out = power_trans @ P_in
            return jnp.mean((P_out - P_target)**2)

        return StateRouter._run_vmap_optimization(engine, loss_fn, num_restarts, max_iters)

    @staticmethod
    def optimize_quantum_routing_vmap(engine, input_occupation, P_target_per_port, 
                                      num_restarts=_QUANTUM_RESTARTS, 
                                      max_iters=_QUANTUM_ITERS):
        """
        Quantum-aware optimizer. Directly optimizes MZI phases to match target
        marginal photon detection probabilities at each output port.
        
        The loss function computes Fock state probabilities via Ryser permanents
        and derives per-port marginal detection probabilities.
        
        Args:
            engine: Engine instance
            input_occupation: list of photon counts per input port, e.g. [1,1,0,...]
            P_target_per_port: desired marginal detection probability per output port
            num_restarts: parallel optimization restarts (reduced due to heavier cost)
            max_iters: Adam steps per restart
        """
        n_modes = engine.n_modes
        n_photons = sum(input_occupation)
        
        if n_photons == 0:
            return []
        
        # Build input indices (repeated for boson occupation)
        in_indices = []
        for mode, count in enumerate(input_occupation):
            in_indices.extend([mode] * count)
        in_indices = jnp.array(in_indices, dtype=jnp.int32)
        
        # Build output Fock basis
        basis = []
        for p in itertools.combinations_with_replacement(range(n_modes), n_photons):
            state = [0] * n_modes
            for m in p:
                state[m] += 1
            basis.append(tuple(state))
        
        out_indices_list = []
        for out_state in basis:
            out_idx = []
            for mode, count in enumerate(out_state):
                out_idx.extend([mode] * count)
            out_indices_list.append(out_idx)
        out_indices_arr = jnp.array(out_indices_list, dtype=jnp.int32)
        
        # Normalization constants
        in_factorials = jnp.array([float(_factorial(n)) for n in input_occupation])
        norm_in = jnp.prod(in_factorials)
        out_factorials = jnp.array([[float(_factorial(n)) for n in state] for state in basis])
        norm_out = jnp.prod(out_factorials, axis=1)
        
        # Build basis-to-port mapping: for each basis state, which ports have photons
        # We compute marginals: P(port k) = sum of probs of all states where port k has >= 1 photon
        basis_arr = jnp.array(basis, dtype=jnp.float64)  # (n_basis, n_modes)
        port_mask = (basis_arr >= 1).astype(jnp.float64)  # (n_basis, n_modes)
        
        # Target marginal probabilities
        P_target = jnp.array(P_target_per_port, dtype=jnp.float64)
        target_sum = jnp.sum(P_target)
        if target_sum > 0:
            P_target = P_target / target_sum * n_photons  # normalize to n_photons total expected
        
        n_mzis = len(engine.mzi_ids)
        
        def loss_fn(phases):
            thetas = jnp.mod(phases[:n_mzis], 2 * jnp.pi)
            phis = jnp.mod(phases[n_mzis:], 2 * jnp.pi)
            U = engine.compute_full_unitary(thetas, phis)
            
            # Compute Fock probabilities via permanents
            all_U_sub = U[out_indices_arr][:, :, in_indices]
            all_perms = jax.lax.map(glynn_permanent, all_U_sub)
            all_perm_sq = jnp.abs(all_perms) ** 2
            all_probs = all_perm_sq / (norm_in * norm_out)
            
            # Compute marginal detection probabilities per port
            # P_marginal[k] = sum of (n_k * prob) for each basis state, where n_k = occupation of port k
            # This gives the expected photon number at each port
            P_marginal = basis_arr.T @ all_probs  # (n_modes,)
            
            return jnp.mean((P_marginal - P_target)**2)
        
        return StateRouter._run_vmap_optimization(engine, loss_fn, num_restarts, max_iters)
