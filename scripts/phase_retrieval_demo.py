import numpy as np
from scipy.stats import unitary_group

def get_unitary_mapping_state_to_basis(psi):
    """
    Returns a unitary V such that V * |0> = psi
    psi is an N-dimensional complex vector (normalized).
    """
    N = len(psi)
    psi = np.asarray(psi, dtype=np.complex128)
    # We want V[:, 0] = psi
    # We can use QR decomposition or just Gram-Schmidt
    # To use QR, we can put psi as the first column of a random matrix
    # But an easier way is Householder or just extending the basis
    
    # Let's just create a random matrix with psi as the first column, then QR
    mat = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    mat[:, 0] = psi
    Q, R = np.linalg.qr(mat)
    # Q[:, 0] will be psi * (R[0,0] / abs(R[0,0]))
    # We just need to adjust the phase of the first column
    phase = R[0, 0] / np.abs(R[0, 0])
    Q[:, 0] = Q[:, 0] * phase
    
    # Let's check
    assert np.allclose(Q[:, 0], psi)
    return Q

def generate_routing_unitaries(psi_in, psi_out, num_unitaries=10):
    """
    Generates `num_unitaries` different unitary matrices U such that
    U @ psi_in = psi_out.
    """
    N = len(psi_in)
    V_in = get_unitary_mapping_state_to_basis(psi_in)
    V_out = get_unitary_mapping_state_to_basis(psi_out)
    
    unitaries = []
    for _ in range(num_unitaries):
        # Generate random (N-1)x(N-1) unitary
        if N > 1:
            U_sub = unitary_group.rvs(N - 1)
        else:
            U_sub = np.eye(0)
            
        # Construct block diagonal matrix
        W_inner = np.eye(N, dtype=np.complex128)
        if N > 1:
            W_inner[1:, 1:] = U_sub
            
        # U = V_out * W_inner * V_in^dagger
        U = V_out @ W_inner @ V_in.conj().T
        unitaries.append(U)
        
    return unitaries

if __name__ == "__main__":
    N = 8
    # Random normalized input state
    psi_in = np.random.randn(N) + 1j * np.random.randn(N)
    psi_in /= np.linalg.norm(psi_in)
    
    # Random normalized output state
    psi_out = np.random.randn(N) + 1j * np.random.randn(N)
    psi_out /= np.linalg.norm(psi_out)
    
    unitaries = generate_routing_unitaries(psi_in, psi_out, num_unitaries=10)
    
    for i, U in enumerate(unitaries):
        out = U @ psi_in
        # Check if it matches psi_out
        fidelity = np.abs(np.vdot(psi_out, out))**2
        print(f"Unitary {i+1}: fidelity = {fidelity:.6f}, is_unitary = {np.allclose(U.conj().T @ U, np.eye(N))}")
