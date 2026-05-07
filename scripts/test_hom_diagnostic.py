"""Diagnostic: HOM routing scenarios — what's physically possible vs what the solver finds."""
import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

import numpy as np
import jax.numpy as jnp
from src.engine import Engine
from src.routing import StateRouter

engine = Engine(n_modes=8)

def test_scenario(name, input_occ, P_target):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  Input:  {input_occ}")
    print(f"  Target: {P_target}")
    print(f"{'='*70}")
    
    results = StateRouter.optimize_quantum_routing_vmap(engine, input_occ, P_target)
    
    print(f"  Found {len(results)} unique solutions")
    for i, (thetas, phis, loss) in enumerate(results[:3]):
        probs, basis = engine.propagate_fock(thetas, phis, input_occ)
        probs = np.asarray(probs)
        
        # Compute marginals (expected photon count per port)
        basis_arr = np.array(basis, dtype=np.float64)
        marginals = basis_arr.T @ probs
        
        print(f"\n  Option {i+1}: loss={loss:.8f}")
        print(f"    Marginal photons per port: {np.round(marginals, 4)}")
        print(f"    Top Fock states:")
        sorted_idx = np.argsort(-probs)
        for j in sorted_idx[:5]:
            if probs[j] > 0.005:
                print(f"      |{''.join(map(str, basis[j]))}> : {probs[j]*100:.2f}%")

# ═══════════════════════════════════════════════════════════════════════
# Test 1: Classic HOM — 1+1 at a 50:50 BS, expect bunching
# ═══════════════════════════════════════════════════════════════════════
test_scenario(
    "Classic HOM: 1+1 in ports 1,2 → expect bunching in ports 1,2",
    [1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0]
)

# ═══════════════════════════════════════════════════════════════════════
# Test 2: Route 1+1 to DIFFERENT output ports (should work perfectly)
# ═══════════════════════════════════════════════════════════════════════
test_scenario(
    "Route 1+1 in ports 1,2 → one photon each to ports 5,6",
    [1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0]
)

# ═══════════════════════════════════════════════════════════════════════
# Test 3: Route 1+1 to SAME port (physically limited!)
# ═══════════════════════════════════════════════════════════════════════
test_scenario(
    "Route 1+1 in ports 1,2 → both to port 5 (HARD - physics limit)",
    [1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0]
)

# ═══════════════════════════════════════════════════════════════════════
# Test 4: Single photon routing (should always work perfectly)
# ═══════════════════════════════════════════════════════════════════════
test_scenario(
    "Single photon: port 1 → port 5",
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0]
)

# ═══════════════════════════════════════════════════════════════════════
# Test 5: 2 photons same input → move to different port
# ═══════════════════════════════════════════════════════════════════════
test_scenario(
    "2 photons same port: port 1 → port 5",
    [2, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0]
)

# ═══════════════════════════════════════════════════════════════════════
# Test 6: Route 1+1 to ports 3,7 (far apart, should work)
# ═══════════════════════════════════════════════════════════════════════
test_scenario(
    "Route 1+1 in ports 1,2 → ports 3 and 7",
    [1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0]
)

# ═══════════════════════════════════════════════════════════════════════
# Test 7: Route 1+1 from non-adjacent inputs
# ═══════════════════════════════════════════════════════════════════════
test_scenario(
    "Route 1+1 in ports 1,5 → ports 3,7",
    [1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0]
)
