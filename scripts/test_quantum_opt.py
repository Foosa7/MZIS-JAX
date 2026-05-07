"""Test: quantum-aware optimizer for multi-photon routing."""
import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

import numpy as np
from src.engine import Engine
from src.routing import StateRouter

engine = Engine(n_modes=4)

print("=== Test 1: 1 photon in port 1, target port 3 ===")
input_occ = [1, 0, 0, 0]
P_target = [0, 0, 1, 0]

results = StateRouter.optimize_quantum_routing_vmap(engine, input_occ, P_target, num_restarts=100, max_iters=150)
print(f"Found {len(results)} unique solutions")
for i, (thetas, phis, loss) in enumerate(results[:3]):
    probs, basis = engine.propagate_fock(thetas, phis, input_occ)
    probs = np.asarray(probs)
    print(f"  Option {i+1}: loss={loss:.8f}")
    for j, (state, p) in enumerate(zip(basis, probs)):
        if p > 0.01:
            print(f"    |{''.join(map(str,state))}> : {p:.4f}")

print("\n=== Test 2: 2 photons (1+1) in ports 1,2, target both to port 3 ===")
input_occ2 = [1, 1, 0, 0]
P_target2 = [0, 0, 1, 0]

results2 = StateRouter.optimize_quantum_routing_vmap(engine, input_occ2, P_target2, num_restarts=100, max_iters=150)
print(f"Found {len(results2)} unique solutions")
for i, (thetas, phis, loss) in enumerate(results2[:3]):
    probs, basis = engine.propagate_fock(thetas, phis, input_occ2)
    probs = np.asarray(probs)
    print(f"  Option {i+1}: loss={loss:.8f}")
    for j, (state, p) in enumerate(zip(basis, probs)):
        if p > 0.01:
            print(f"    |{''.join(map(str,state))}> : {p:.4f}")

print("\n=== Test 3: HOM - 2 photons (1+1), target 50/50 bunching ===")
input_occ3 = [1, 1, 0, 0]
P_target3 = [1, 1, 0, 0]  # expect photons to bunch into ports 1 or 2

results3 = StateRouter.optimize_quantum_routing_vmap(engine, input_occ3, P_target3, num_restarts=100, max_iters=150)
print(f"Found {len(results3)} unique solutions")
for i, (thetas, phis, loss) in enumerate(results3[:3]):
    probs, basis = engine.propagate_fock(thetas, phis, input_occ3)
    probs = np.asarray(probs)
    print(f"  Option {i+1}: loss={loss:.8f}")
    for j, (state, p) in enumerate(zip(basis, probs)):
        if p > 0.01:
            print(f"    |{''.join(map(str,state))}> : {p:.4f}")
