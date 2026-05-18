import numpy as np
import jax
import jax.numpy as jnp
import optax
from src.engine import Engine

# Detect GPU availability and set performance tiers
_HAS_GPU = jax.default_backend() == 'gpu'

# GPU: push hard — plenty of VRAM and parallelism
# CPU: conservative — avoid hogging the main thread for too long
if _HAS_GPU:
    _RESTARTS = 100
    _ITERS = 500
    _LR = 0.03
else:
    _RESTARTS = 20
    _ITERS = 200
    _LR = 0.05

print(f"[UnitaryOptimizer] Backend: {jax.default_backend()} | "
      f"Restarts: {_RESTARTS} | Iters: {_ITERS} | LR: {_LR}")

class UnitaryOptimizer:
    
    @staticmethod
    def _run_vmap_optimization(engine: Engine, loss_fn, num_restarts, max_iters, n_mzis, n_modes):
        """
        Core vmap optimization engine. Takes a loss_fn(phases) and runs
        num_restarts parallel Adam optimizations, returning the best phase configuration.
        """
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
        # We need phases for thetas, phis, and alphas
        init_phases_batch = jax.random.uniform(
            key, shape=(num_restarts, 2 * n_mzis + n_modes), minval=0, maxval=2*jnp.pi
        )

        final_phases_batch, final_losses = parallel_optimization(init_phases_batch)

        # Fallback: return the single best run
        best_idx = jnp.argmin(final_losses)
        best_phases = final_phases_batch[best_idx]
        best_loss = final_losses[best_idx]
        
        best_phases = np.mod(best_phases, 2*np.pi)
        thetas = best_phases[:n_mzis]
        phis = best_phases[n_mzis:2*n_mzis]
        alphas = best_phases[2*n_mzis:]
        
        return np.array(thetas), np.array(phis), np.array(alphas), float(best_loss)


    @staticmethod
    def optimize_unitary_vmap(engine: Engine, U_target: np.ndarray, 
                                num_restarts=_RESTARTS, 
                                max_iters=_ITERS):
        """
        Directly optimizes the physical hardware phases to synthesize a target Unitary matrix.
        Note: Physical chips usually lack the final layer of diagonal phase screens (alphas)
        required by Clements decomposition. This optimizer simulates them virtually to match 
        the expected target unitary from the actual output exactly.
        """
        U_target = jnp.array(U_target, dtype=jnp.complex128)
        n_mzis = len(engine.mzi_ids)
        n_modes = engine.n_modes

        def loss_fn(phases):
            # first n_mzis are thetas, next n_mzis are phis, last n_modes are alphas
            thetas = jnp.mod(phases[:n_mzis], 2 * jnp.pi)
            phis = jnp.mod(phases[n_mzis:2*n_mzis], 2 * jnp.pi)
            alphas = jnp.mod(phases[2*n_mzis:], 2 * jnp.pi)
            
            U_chip = engine.compute_full_unitary(thetas, phis)
            D_alpha = jnp.diag(jnp.exp(1j * alphas))
            U_full = D_alpha @ U_chip
            
            # Minimize the squared Frobenius distance between actual and expected unitaries
            return jnp.mean(jnp.abs(U_full - U_target)**2)

        return UnitaryOptimizer._run_vmap_optimization(engine, loss_fn, num_restarts, max_iters, n_mzis, n_modes)
