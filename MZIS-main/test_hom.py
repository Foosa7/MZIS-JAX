import jax.numpy as jnp
from src.engine import Engine

e = Engine(n_modes=4)
e.phases[e.layout[0][0]['id']]['theta'] = float(jnp.pi / 2)
res, _ = e.propagate_fock([1, 1, 0, 0])
print("Probabilities:", res)
