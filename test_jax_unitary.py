import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import unitary_group
import sys
sys.path.append('.')
from src.engine import Engine
from src.routing import StateRouter

np.random.seed(42)
N = 4
U_target = unitary_group.rvs(N)
engine = Engine(n_modes=N, bs_error=0.0)

# We will optimize the phases to match U_target
n_mzis = len(engine.mzi_ids)

def loss_fn(phases):
    thetas = jnp.mod(phases[:n_mzis], 2 * jnp.pi)
    phis = jnp.mod(phases[n_mzis:2*n_mzis], 2 * jnp.pi)
    # the chip lacks output phase screens, so we will optimize them too just for matching!
    alphas = jnp.mod(phases[2*n_mzis:], 2 * jnp.pi)
    
    U_chip = engine.compute_full_unitary(thetas, phis)
    
    # Apply output phase screen
    D_alpha = jnp.diag(jnp.exp(1j * alphas))
    U_full = D_alpha @ U_chip
    
    return jnp.mean(jnp.abs(U_full - U_target)**2)

# Run optimization
import optax
optimizer = optax.adam(learning_rate=0.03)

def single_optimization(init_phases):
    opt_state = optimizer.init(init_phases)
    def step(carry, _):
        params, state = carry
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
        return (params, state), loss_val
    (final_params, _), loss_history = jax.lax.scan(step, (init_phases, opt_state), None, length=1000)
    return final_params, loss_history[-1]

parallel_optimization = jax.jit(jax.vmap(single_optimization))
key = jax.random.PRNGKey(42)
init_phases_batch = jax.random.uniform(key, shape=(200, 2 * n_mzis + N), minval=0, maxval=2*jnp.pi)

final_phases, final_losses = parallel_optimization(init_phases_batch)
best_idx = jnp.argmin(final_losses)
best_loss = final_losses[best_idx]
print(f"Best reconstruction loss with physical chip + alphas: {best_loss:.10f}")
