import jax.numpy as jnp
import numpy as np
import sys
sys.path.append('/home/foosa/jax-env')
from src.engine import Engine
from decompose.pnn import decompose_clements

N = 8
engine = Engine(n_modes=N)

# Generate random Haar unitary
Z = np.random.randn(N, N) + 1j * np.random.randn(N, N)
Q, R = np.linalg.qr(Z)
d = np.diagonal(R)
ph = d / np.abs(d)
U_target = Q @ np.diag(ph)

phis, thetas, alphas = decompose_clements(U_target, block='mzi')

phases = {}
for mid in engine.mzi_ids:
    phases[mid] = {'theta': np.pi, 'phi': 0.0}

for col_idx, col in enumerate(engine.layout):
    p = col_idx // 2
    for mzi in col:
        q = mzi['mode_top']
        mid = mzi['id']
        phases[mid]['theta'] = float(np.mod(2 * thetas[q, p], 2 * np.pi))
        phases[mid]['phi'] = float(np.mod(phis[q, p], 2 * np.pi))

t_arr = jnp.array([phases[mid]['theta'] for mid in engine.mzi_ids])
p_arr = jnp.array([phases[mid]['phi'] for mid in engine.mzi_ids])

U_chip = engine.compute_full_unitary(t_arr, p_arr)

P_meas = np.abs(U_chip)**2
P_exp = np.abs(U_target)**2

prod = P_exp * P_meas
sum_overlap = np.sum(np.sqrt(np.clip(prod, 0, None)))
fidelity = sum_overlap / N

print(f"Fidelity: {fidelity:.6f}")
