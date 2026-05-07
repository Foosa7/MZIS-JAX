import jax.numpy as jnp
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.engine import Engine

def test_hardware_calibration_mapping():
    print("=== End-to-End Hardware Calibration Test ===")
    
    # 1. We start with an imbalanced input beam that we want to nullify at a specific node (e.g. G4_theta)
    # The beamsplitter has a known defect (e.g. e=0.05)
    input_amps = jnp.array([1.0, 0.3j])
    e_l, e_r = 0.05, 0.05
    
    # 2. Parallel Nullification: Calculate the EXACT theoretical phase needed
    theta_req, phi_req = Engine.parallel_nullification(input_amps, e_l, e_r)
    print(f"1. Nullification target:")
    print(f"   Theoretical Theta: {theta_req/jnp.pi:.3f}π, Phi: {phi_req/jnp.pi:.3f}π")
    
    # 3. Phase Constraints: Optimize the phase for the hardware (avoid excessive heating)
    theta_opt, phi_opt = Engine.apply_phase_constraints(jnp.array([theta_req]), jnp.array([phi_req]))
    theta_target = float(theta_opt[0])
    print(f"2. Hardware constraints applied:")
    print(f"   Optimized Theta: {theta_target/jnp.pi:.3f}π")
    
    # 4. Hardware Mapping: Translate the optical phase to electrical current
    # Load the 8-mode digital twin auto-cal data
    json_path = os.path.join(os.path.dirname(__file__), '..', 'node-isolation', '8-mode-autocal-20260209.json')
    with open(json_path, 'r') as f:
        cal_data = json.load(f)
        
    heater_id = "G4_theta"
    phase_params = cal_data['phase_calibration'][heater_id]['phase_params']
    res_params = cal_data['resistance_calibration'][heater_id]['resistance_params']
    
    # In your digital twin, phase = omega * Electrical_Power + offset
    # Therefore, Required Power = Theta / omega
    omega = phase_params['omega']
    req_power = theta_target / omega
    
    # Now we invert the thermo-optic resistance model: P = I^2 * R(I)
    # R(I) = c_res + a_res * I^2 + d_res * I^4
    # P = I^2 * (c_res + a_res * I^2 + d_res * I^4) = c_res*I^2 + a_res*I^4 + d_res*I^6
    c_res = res_params['c_res']
    a_res = res_params['a_res']
    d_res = res_params['d_res']
    
    # We can solve for I^2 by finding the roots of the polynomial: d*x^3 + a*x^2 + c*x - P = 0 (where x = I^2)
    roots = np.roots([d_res, a_res, c_res, -req_power])
    
    # Find the real, positive root for I^2
    valid_roots = [r.real for r in roots if np.isreal(r) and r.real > 0]
    i_squared = valid_roots[0]
    i_req = np.sqrt(i_squared)
    
    print(f"3. Digital Twin Hardware Mapping for {heater_id}:")
    print(f"   Required Electrical Power: {req_power:.3f} mW")
    print(f"   Required DAC Current:      {i_req:.3f} mA")
    print(f"\nSUCCESS: You can now route light directly by sending {i_req:.3f} mA to pin {cal_data['phase_calibration'][heater_id]['pin']}!")

if __name__ == "__main__":
    test_hardware_calibration_mapping()
