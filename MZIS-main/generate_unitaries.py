import numpy as np
import os

def generate_random_unitary(dim):
    """Generates a random unitary matrix following the Haar measure."""
    Z = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    Q, R = np.linalg.qr(Z)
    d = np.diagonal(R)
    ph = d / np.abs(d)
    return Q @ np.diag(ph)

if __name__ == "__main__":
    # Generate random unitaries for N=8, 12, 16 modes
    dims = [8, 12, 16]
    
    output_dir = os.path.abspath(os.path.dirname(__file__))
    
    for dim in dims:
        U = generate_random_unitary(dim)
        filename = os.path.join(output_dir, f"random_unitary_{dim}x{dim}.npy")
        np.save(filename, U)
        print(f"Successfully saved {filename}")
