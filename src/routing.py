import numpy as np
from scipy.stats import unitary_group

class StateRouter:
    @staticmethod
    def get_unitary_mapping_state_to_basis(psi):
        """
        Returns a unitary V such that V * |0> = psi
        psi is an N-dimensional complex vector (normalized).
        """
        N = len(psi)
        psi = np.asarray(psi, dtype=np.complex128)
        
        # QR decomposition to complete the basis
        mat = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        mat[:, 0] = psi
        Q, R = np.linalg.qr(mat)
        
        # Adjust phase of the first column so Q[:, 0] exactly matches psi
        phase = R[0, 0] / np.abs(R[0, 0])
        Q[:, 0] = Q[:, 0] * phase
        
        return Q

    @staticmethod
    def generate_routing_unitaries(psi_in, psi_out, num_unitaries=10):
        """
        Generates `num_unitaries` different unitary matrices U such that
        U @ psi_in = psi_out (up to a global phase or exactly).
        """
        N = len(psi_in)
        V_in = StateRouter.get_unitary_mapping_state_to_basis(psi_in)
        V_out = StateRouter.get_unitary_mapping_state_to_basis(psi_out)
        
        unitaries = []
        for _ in range(num_unitaries):
            if N > 1:
                U_sub = unitary_group.rvs(N - 1)
            else:
                U_sub = np.eye(0)
                
            W_inner = np.eye(N, dtype=np.complex128)
            if N > 1:
                W_inner[1:, 1:] = U_sub
                
            U = V_out @ W_inner @ V_in.conj().T
            unitaries.append(U)
            
        return unitaries
