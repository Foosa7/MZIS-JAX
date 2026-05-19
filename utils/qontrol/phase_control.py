# app/utils/qontrol/phase_control.py

import json
import copy
import logging

import jax.numpy as jnp
from jax import jit

import numpy as np
from scipy.optimize import brentq
from app.utils.qontrol.mapping_utils import get_mapping_functions
from app.utils.appdata import AppData

class PhaseController:
    '''
    Utility class for controlling the phases.
    '''

    @staticmethod
    def solve_current_analytical_numpy(P_mW_array, c_res_array, alpha_res_array, max_current_mA=6.1):
        """
        Pure NumPy Vectorized Analytical Solver with Hardware Guardrails.
        Instantly computes all 64 currents while protecting the TiN heaters.
        
        Args:
            P_mW_array: Target power in milliwatts.
            c_res_array: Base resistance values.
            alpha_res_array: Thermal non-linearity coefficients.
            max_current_mA: Hard hardware limit (defaults to 6.1 mA from original brentq bounds).
        """
        # EDGE CASE 1: Negative power requests
        # Physically impossible to require negative heating power. Clamp to 0.
        P_safe = np.maximum(0.0, P_mW_array)
        
        # EDGE CASE 2: Perfectly linear resistors
        # If alpha is exactly 0, the quadratic formula divides by zero. 
        # We substitute a microscopic epsilon to prevent NaN errors in the array math.
        safe_alpha = np.where(alpha_res_array == 0, 1e-12, alpha_res_array)
        
        # The Quadratic Formula
        discriminant = 1.0 + 4.0 * safe_alpha * (P_safe / c_res_array)
        I_squared = (-1.0 + np.sqrt(discriminant)) / (2.0 * safe_alpha)
        
        # Resolve EDGE CASE 2
        # If alpha was originally 0, overwrite the quadratic result with simple Ohm's law.
        I_squared_final = np.where(alpha_res_array == 0, P_safe / c_res_array, I_squared)
        
        # EDGE CASE 3: Floating point underflow
        # If I_squared evaluates to -0.0000000001, np.sqrt() will throw a complex number warning.
        I_raw = np.sqrt(np.maximum(0.0, I_squared_final))
        
        # EDGE CASE 4: Hardware protection limit
        # Prevent the DACs from sending currents high enough to melt the waveguides.
        I_clipped = np.clip(I_raw, 0.0, max_current_mA)
        
        # EDGE CASE 5: Stray NaNs
        # If calibration data was missing and c_res was 0, it creates a NaN. 
        # Safely ground that specific channel to 0.0 mA instead of crashing the server.
        I_final = np.nan_to_num(I_clipped, nan=0.0)
        
        return I_final


    # @staticmethod
    # def solve_current_with_brentq(P_mW, c_res, alpha_res):
    #     '''
    #     Solve for current I given Power P using Brent's method.
    #     Equation: I^2 * (1 + alpha * I^2) - (P / R_0) = 0
    #     '''
    #     def equation(I):
    #         return I**2 * (1 + alpha_res * I**2) - (P_mW / c_res)
    #     try:
    #         return brentq(equation, a=1e-5, b=6.1, maxiter=100)
    #     except ValueError as e:
    #         logging.error(f"brentq failed to find a root: {e}")
    #         return None

    @classmethod
    def calculate_current_for_phase(cls, calib_key, phase_value):
        '''
        Calculate required current for a specific phase value.
        '''
        # Skip uncalibrated keys
        SKIP_KEYS = {
            "A1_phi", "A2_phi", "A3_phi", "A4_phi", "A5_phi", "A6_phi",
            "B1_phi", "B2_phi", "B3_phi", "B4_phi", "B5_phi"
        }
        if calib_key in SKIP_KEYS:
            return None
        
        res_cal = AppData.resistance_calibration_data.get(calib_key)
        phase_cal = AppData.phase_calibration_data.get(calib_key)

        if res_cal is None or phase_cal is None:
            logging.error(f"Missing calibration for {calib_key}")
            return None

        res_params = res_cal.get("resistance_params")
        phase_params = phase_cal.get("phase_params")
        if res_params is None or phase_params is None:
            logging.error(f"Missing calibration params for {calib_key}")
            return None
        try:
            c_res = res_params.get('c_res')
            a_res = res_params.get('a_res')
            alpha_res = res_params.get('alpha_res')
            A = phase_params.get('amplitude')
            b = phase_params.get('omega')
            c = phase_params.get('phase')
            d = phase_params.get('offset')
            if None in (c_res, a_res, alpha_res, A, b, c, d):
                logging.error(f"Missing parameter value for {calib_key}")
                return None
        except Exception as e:
            logging.error(f"Failed to extract parameters for {calib_key}: {e}")
            return None

        delta_phase = (phase_value - c) % 2
        P_mW = delta_phase * np.pi / b  # Power in mW
        return cls.solve_current_with_brentq(P_mW, c_res, alpha_res)

    @classmethod
    def apply_phases_to_hardware(cls, qontrol, grid_size, grid_config):
        '''
        Apply a full grid phase configuration to hardware.
        '''
        create_label_mapping, apply_grid_mapping = get_mapping_functions(grid_size)
        label_map = create_label_mapping(int(grid_size.split('x')[0]))

        phase_grid_config = copy.deepcopy(grid_config)
        applied_channels = []
        failed_channels = []

        # Process each cross in the grid
        for cross_label, data in grid_config.items():
            if cross_label not in label_map:
                continue

            # Process theta value
            theta_val = data.get("theta", "0")
            if theta_val:
                try:
                    theta_float = float(theta_val)
                    calib_key = f"{cross_label}_theta"
                    current_theta = cls.calculate_current_for_phase(calib_key, theta_float)
                    if current_theta is not None:
                        current_theta = round(current_theta, 5)
                        phase_grid_config[cross_label]["theta"] = str(current_theta)
                        applied_channels.append(f"{cross_label}:θ = {current_theta:.5f} mA")
                    else:
                        failed_channels.append(f"{cross_label}:θ (no calibration)")
                except Exception as e:
                    failed_channels.append(f"{cross_label}:θ ({str(e)})")

            # Process phi value
            phi_val = data.get("phi", "0")
            if phi_val:
                try:
                    phi_float = float(phi_val)
                    channel = f"{cross_label}_phi"
                    current_phi = cls.calculate_current_for_phase(channel, phi_float)
                    
                    if current_phi is not None:
                        current_phi = round(current_phi, 5)
                        phase_grid_config[cross_label]["phi"] = str(current_phi)
                        applied_channels.append(f"{cross_label}:φ = {current_phi:.5f} mA")
                    else:
                        failed_channels.append(f"{cross_label}:φ (no calibration)")
                except Exception as e:
                    failed_channels.append(f"{cross_label}:φ ({str(e)})")

        # Apply configuration to hardware
        try:
            config_json = json.dumps(phase_grid_config)
            apply_grid_mapping(qontrol, config_json, grid_size)
        except Exception as e:
            logging.error(f"Hardware application failed: {e}")
            failed_channels.append(f"Hardware write failed: {e}")
        return phase_grid_config, applied_channels, failed_channels

    @staticmethod
    def create_zero_config(grid_size):
        '''
        Create a configuration with zero phases for all heaters.
        '''
        n = int(grid_size.split('x')[0])
        zero_config = {}
        
        # Generate all possible crosspoint labels (A1, A2, B1, etc.)
        for row in range(n):
            row_letter = chr(65 + row)  # A, B, C, etc.
            for col in range(1, n+1):
                cross_label = f"{row_letter}{col}"
                zero_config[cross_label] = {
                    "arms": ["TL", "TR", "BL", "BR"],  # Include all arms
                    "theta": "0",
                    "phi": "0"
                }       
        return json.dumps(zero_config)
