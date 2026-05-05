import jax.numpy as jnp
from src.engine import Engine

e = Engine(n_modes=4)

thetas = jnp.full(e.n_mzis, jnp.pi)
phis = jnp.zeros(e.n_mzis)

# Set first MZI (A1) to a 50:50 beam splitter
thetas = thetas.at[0].set(jnp.pi / 2)

probs, basis = e.propagate_fock(thetas, phis, [1, 1, 0, 0])

for i, b in enumerate(basis):
    if float(probs[i]) > 0.0001:
        print(f"{b}: {probs[i]:.4f}")
