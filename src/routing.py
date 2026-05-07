import numpy as np
import jax
import jax.numpy as jnp
import optax
from scipy.stats import unitary_group
from scipy.optimize import minimize

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
    def optimize_incoherent_routing_vmap(engine, P_in, P_target, num_restarts=1000, max_iters=200):
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

        optimizer = optax.adam(learning_rate=0.05)

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

        success_threshold = 1e-4
        successful_mask = final_losses < success_threshold
        
        # JAX arrays to NumPy for easy dynamic lists
        valid_phases = np.array(final_phases_batch[successful_mask])
        valid_losses = np.array(final_losses[successful_mask])

        unique_options = []
        unique_losses = []
        tolerance = 1e-2

        for p, l in zip(valid_phases, valid_losses):
            p_wrapped = np.mod(p, 2 * np.pi)
            is_new = True
            for unique_p in unique_options:
                # Wrap difference to handle 0 and 2pi equivalence
                diff = np.abs(p_wrapped - unique_p)
                diff = np.minimum(diff, 2*np.pi - diff)
                if np.mean(diff) < tolerance:
                    is_new = False
                    break
                    
            if is_new:
                unique_options.append(p_wrapped)
                unique_losses.append(l)
                
        # If no runs were completely successful, at least return the best one
        if not unique_options:
            best_idx = jnp.argmin(final_losses)
            best_phases = np.array(final_phases_batch[best_idx])
            best_loss = final_losses[best_idx]
            unique_options.append(np.mod(best_phases, 2*np.pi))
            unique_losses.append(best_loss)

        # Convert back to (thetas, phis, loss) tuples
        results = []
        for p, l in zip(unique_options, unique_losses):
            thetas = p[:n_mzis]
            phis = p[n_mzis:]
            results.append((thetas, phis, l))
            
        return results
