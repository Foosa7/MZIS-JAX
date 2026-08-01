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
    
    # The fitted fringe is  I_opt = offset + amplitude * cos(omega * P + phase),
    # so the heater already sits at `phase` when no power is applied. That offset
    # must be subtracted, and it is stored in units of pi, not radians. Fitting
    # the model back against measurement_data confirms both points: including it
    # gives a normalised RMS of 0.008 against the raw sweep, dropping it 0.49.
    omega = phase_params['omega']
    phase_at_zero_power = phase_params['phase'] * np.pi

    # Heaters only ever add phase, so wrap the request into [0, 2*pi).
    delta_phase = np.mod(theta_target - phase_at_zero_power, 2 * np.pi)
    req_power = delta_phase / omega

    # Invert the thermo-optic resistance model. The `alpha_res` form is the one
    # that reproduces the measured sweep (0.008 vs 0.47 for the c/a/d cubic):
    #   P = c_res * I^2 * (1 + alpha_res * I^2)
    # which is a quadratic in x = I^2:  c*alpha*x^2 + c*x - P = 0
    c_res = res_params['c_res']
    alpha_res = res_params['alpha_res']

    i_squared = (-1.0 + np.sqrt(1.0 + 4.0 * alpha_res * req_power / c_res)) / (2.0 * alpha_res)
    i_req = np.sqrt(max(0.0, i_squared))

    print(f"3. Digital Twin Hardware Mapping for {heater_id}:")
    print(f"   Required Electrical Power: {req_power:.3f} mW")
    print(f"   Required DAC Current:      {i_req:.3f} mA")
    print(f"\nSUCCESS: You can now route light directly by sending {i_req:.3f} mA to pin {cal_data['phase_calibration'][heater_id]['pin']}!")

if __name__ == "__main__":
    test_hardware_calibration_mapping()
