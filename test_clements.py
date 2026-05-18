import numpy as np
import jax.numpy as jnp
from scipy.stats import unitary_group
import sys
sys.path.append('.')
from src.engine import Engine
from decompose.pnn import decompose_clements, reconstruct_clements

np.random.seed(42)
N = 4
U_target = unitary_group.rvs(N)

# 1. Decompose using PNN Clements
phis, thetas, alphas = decompose_clements(U_target, block='mzi')

# 2. Reconstruct using PNN Clements
U_pnn = reconstruct_clements(phis, thetas, alphas, block='mzi')

print("PNN Reconstruction error:", np.linalg.norm(U_pnn - U_target))

# 3. Try to map to Engine physical MZI
engine = Engine(n_modes=N, bs_error=0.0)

print(f"Phases shape: {phis.shape}")
print("Thetas:\n", thetas)
print("Phis:\n", phis)
print("Alphas:\n", alphas)

# Let's see how many MZIs there are in engine.
n_mzis = len(engine.mzi_ids)
print("Engine MZIs:", n_mzis)

# Engine layout is even-odd alternating.
# PNN applies p=0 (col 0): q=0, 2 (even), then q=1, 3 (odd)
# For N=4, row=3, col=2.
# Even layers (q=0, 2): (0,1), (2,3)
# Odd layers (q=1, 3): (1,2) [q=3 out of bounds for dim 4]
# PNN order of applying (right to left on vector):
# p=0, q=0 (modes 0,1)
# p=0, q=2 (modes 2,3)
# p=0, q=1 (modes 1,2)
# p=1, q=0 (modes 0,1)
# p=1, q=2 (modes 2,3)
# p=1, q=1 (modes 1,2)
# Total 6 MZIs. Engine has exactly N(N-1)/2 MZIs?
# For N=4, Engine has:
# col 0 (even): (0,1), (2,3) -> count = 2
# col 1 (odd): (1,2) -> count = 1
# col 2 (even): (0,1), (2,3) -> count = 2
# col 3 (odd): (1,2) -> count = 1
# Total 6 MZIs. Matches perfectly.

thetas_eng = np.zeros(6)
phis_eng = np.zeros(6)

# Map pnn phases to engine array
# p=0, even
thetas_eng[0] = thetas[0, 0]
phis_eng[0]   = phis[0, 0]
thetas_eng[1] = thetas[2, 0]
phis_eng[1]   = phis[2, 0]
# p=0, odd
thetas_eng[2] = thetas[1, 0]
phis_eng[2]   = phis[1, 0]
# p=1, even
thetas_eng[3] = thetas[0, 1]
phis_eng[3]   = phis[0, 1]
thetas_eng[4] = thetas[2, 1]
phis_eng[4]   = phis[2, 1]
# p=1, odd
thetas_eng[5] = thetas[1, 1]
phis_eng[5]   = phis[1, 1]

# Apply the mathematical mapping: theta_eng = -2*theta_pnn, phi_eng = phi_pnn - pi
thetas_eng_mapped = -2 * thetas_eng
phis_eng_mapped = phis_eng - np.pi

U_eng_mapped = engine.compute_full_unitary(jnp.array(thetas_eng_mapped), jnp.array(phis_eng_mapped))

# Since U_eng_mapped has Z_2 flips, let's conjugate by Z
Z = np.diag([1, -1, 1, -1])
U_eng_Z = Z @ U_eng_mapped @ Z

# Note we also miss the 'alphas' output phase screen.
D_alpha = np.diag(np.exp(1j * alphas))
U_eng_Z_with_alpha = D_alpha @ U_eng_Z

print("Error between target and mapped engine (with Z and alphas):", np.linalg.norm(U_eng_Z_with_alpha - U_target))
