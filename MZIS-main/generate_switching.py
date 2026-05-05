import numpy as np
import os

def generate_random_switching(dim):
    """Generates a random switching (permutation) matrix."""
    P = np.zeros((dim, dim), dtype=np.complex128)
    # Randomly assign each input to a unique output
    cols = np.random.permutation(dim)
    for i in range(dim):
        P[i, cols[i]] = 1.0
    return P

if __name__ == "__main__":
    # Generate random switching matrices for N=8, 12, 16 modes
    dims = [8, 12, 16]
    
    output_dir = os.path.abspath(os.path.dirname(__file__))
    
    for dim in dims:
        P = generate_random_switching(dim)
        filename = os.path.join(output_dir, f"random_switching_{dim}x{dim}.npy")
        np.save(filename, P)
        print(f"Successfully saved {filename}")
