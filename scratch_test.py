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

gamma = np.zeros(N)

for col_idx, col in enumerate(engine.layout):
    p = col_idx // 2
    for mzi in col:
        k = mzi['mode_top']
        mid = mzi['id']
        
        theta_p = thetas[k, p]
        phi_p = phis[k, p]
        
        theta_eng = 2 * theta_p
        phi_eng = phi_p + gamma[k+1] - gamma[k] + np.pi
        
        phases[mid]['theta'] = float(np.mod(theta_eng, 2 * np.pi))
        phases[mid]['phi'] = float(np.mod(phi_eng, 2 * np.pi))
        
        gamma_new = theta_p + gamma[k+1] + np.pi
        gamma[k] = gamma_new
        gamma[k+1] = gamma_new

t_arr = jnp.array([phases[mid]['theta'] for mid in engine.mzi_ids])
p_arr = jnp.array([phases[mid]['phi'] for mid in engine.mzi_ids])

U_chip = engine.compute_full_unitary(t_arr, p_arr)

P_meas = np.abs(U_chip)**2
P_exp = np.abs(U_target)**2

prod = P_exp * P_meas
sum_overlap = np.sum(np.sqrt(np.clip(prod, 0, None)))
fidelity = sum_overlap / N

print(f"Fidelity after correction: {fidelity:.6f}")

# Optional: Verify if U_chip and U_target match up to output phases
U_target_phases = np.angle(U_target)
U_chip_phases = np.angle(U_chip)
