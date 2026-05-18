import numpy as np
import jax.numpy as jnp
import sys
sys.path.append('.')
from src.engine import Engine
from decompose.pnn import U2MZI

dim = 2
engine = Engine(n_modes=dim, bs_error=0.0)

theta = 0.5
phi = 0.3

# engine's MZI
thetas = jnp.array([theta])
phis = jnp.array([phi])
e_ls = jnp.array([0.0])
e_rs = jnp.array([0.0])
mode_tops = jnp.array([0])
U_engine = engine._build_layer_matrix(thetas, phis, e_ls, e_rs, mode_tops, dim)

# pnn's MZI
U_pnn = U2MZI(dim, 0, 1, phi, theta)

print("Engine MZI:")
print(np.array(U_engine))
print("PNN MZI:")
print(U_pnn)
