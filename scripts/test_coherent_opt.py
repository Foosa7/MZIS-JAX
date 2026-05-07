"""Quick test: verify coherent vmap optimizer produces correct routing."""
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

import numpy as np
import jax.numpy as jnp
from src.engine import Engine
from src.routing import StateRouter

engine = Engine(n_modes=8)

# Test: route port 1 -> port 5
psi_in = np.zeros(8, dtype=np.complex128)
psi_in[0] = 1.0

psi_target = np.zeros(8, dtype=np.complex128)
psi_target[4] = 1.0

print("Running coherent vmap optimization (1000 restarts, 200 iters)...")
results = StateRouter.optimize_coherent_routing_vmap(engine, psi_in, psi_target, num_restarts=1000, max_iters=200)

print(f"\nFound {len(results)} unique solutions")
for i, (thetas, phis, loss) in enumerate(results[:5]):
    U = engine.compute_full_unitary(thetas, phis)
    psi_out = np.asarray(U) @ psi_in
    P_out = np.abs(psi_out)**2
    print(f"  Option {i+1}: loss={loss:.8f}, output powers={np.round(P_out, 4)}")
    
print("\n--- Test 2: route port 1 -> split 50/50 to ports 3 and 7 ---")
psi_target2 = np.zeros(8, dtype=np.complex128)
psi_target2[2] = 1.0/np.sqrt(2)
psi_target2[6] = 1.0/np.sqrt(2)

results2 = StateRouter.optimize_coherent_routing_vmap(engine, psi_in, psi_target2, num_restarts=1000, max_iters=200)
print(f"Found {len(results2)} unique solutions")
for i, (thetas, phis, loss) in enumerate(results2[:5]):
    U = engine.compute_full_unitary(thetas, phis)
    psi_out = np.asarray(U) @ psi_in
    P_out = np.abs(psi_out)**2
    print(f"  Option {i+1}: loss={loss:.8f}, output powers={np.round(P_out, 4)}")
