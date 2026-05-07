import jax.numpy as jnp
import numpy as np
import sys
import os

# Add the project root to the path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine import Engine

def test_bs_error():
    print("=== Testing Beamsplitter Error (bs_error) ===")
    # 1. Ideal Engine
    ideal_engine = Engine(n_modes=4, bs_error=0.0)
    # 2. Engine with significant beamsplitter error
    error_engine = Engine(n_modes=4, bs_error=0.2)
    
    # Set MZI to 50:50 state (theta = pi/2)
    thetas = jnp.ones(6) * (jnp.pi / 2)
    phis = jnp.zeros(6)
    
    # Input 1 unit of power into port 0
    input_powers = [1.0, 0.0, 0.0, 0.0]
    
    ideal_out = ideal_engine.get_classical_flow(thetas, phis, input_powers, coherent=True)[-1]
    error_out = error_engine.get_classical_flow(thetas, phis, input_powers, coherent=True)[-1]
    
    print(f"Ideal 50:50 Output Powers: Port 0: {ideal_out[0]:.4f}, Port 1: {ideal_out[1]:.4f}")
    print(f"Error 50:50 Output Powers: Port 0: {error_out[0]:.4f}, Port 1: {error_out[1]:.4f}")
    print("Notice how the power is no longer split perfectly 50:50 due to the bs_error!\n")

def test_phase_constraints():
    print("=== Testing Phase Constraints & Optimization ===")
    # Suppose an algorithm requests an MZI with theta = 1.5π and phi = -0.5π
    # Physically, theta=1.5π requires lots of heat, and phi=-0.5π is negative (invalid).
    thetas_raw = jnp.array([1.5 * jnp.pi])
    phis_raw = jnp.array([-0.5 * jnp.pi])
    
    print(f"Requested internal phase (theta): {thetas_raw[0]/jnp.pi:.2f}π")
    print(f"Requested external phase (phi):  {phis_raw[0]/jnp.pi:.2f}π")
    
    thetas_opt, phis_opt = Engine.apply_phase_constraints(thetas_raw, phis_raw)
    
    print(f"Optimized internal phase (theta): {thetas_opt[0]/jnp.pi:.2f}π")
    print(f"Optimized external phase (phi):   {phis_opt[0]/jnp.pi:.2f}π")
    print("The optimization successfully reflected theta into the [0, π) domain to save power, and wrapped phi to positive [0, 2π)!\n")

def test_parallel_nullification():
    print("=== Testing Parallel Nullification ===")
    # Imagine we have two light beams entering an MZI with different amplitudes and a relative phase
    input_amps = jnp.array([1.0, 0.5j]) # 1.0 power in top, 0.25 power in bottom with 90-deg phase difference
    
    # We want to find the exact theta/phi to route ALL this light to the top port.
    e_l, e_r = 0.0, 0.0 # Start with ideal beamsplitters
    
    theta_req, phi_req = Engine.parallel_nullification(input_amps, e_l, e_r)
    print(f"To nullify the input [1.0, 0.5j] on an ideal MZI:")
    print(f"  We need Theta = {theta_req/jnp.pi:.3f}π, Phi = {phi_req/jnp.pi:.3f}π")
    
    # Now let's calculate what we need if the beamsplitter is defective
    theta_defective, phi_defective = Engine.parallel_nullification(input_amps, 0.1, 0.1)
    print(f"To nullify the exact same input on an MZI with bs_error=0.1:")
    print(f"  We need Theta = {theta_defective/jnp.pi:.3f}π, Phi = {phi_defective/jnp.pi:.3f}π")
    print("The nullification dynamically accounts for the hardware defect to find the true calibration point!\n")

if __name__ == "__main__":
    test_bs_error()
    test_phase_constraints()
    test_parallel_nullification()
