import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from src.engine import Engine

engine = Engine(n_modes=8)
n_mzis = len(engine.mzi_ids)

P_in = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
P_target = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

def loss_fn(phases):
    thetas = phases[:n_mzis]
    phis = phases[n_mzis:]
    U = engine.compute_full_unitary(thetas, phis)
    power_trans = jnp.abs(U)**2
    P_out = power_trans @ P_in
    return jnp.mean((P_out - P_target)**2)

grad_fn = jax.jit(jax.value_and_grad(loss_fn))

phases = np.random.uniform(0, 2*np.pi, 2*n_mzis)
loss, grad = grad_fn(jnp.array(phases))
print(f"Loss: {loss}")
print(f"Grad shape: {grad.shape}")
