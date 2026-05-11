import numpy as np

# 1. Define your file names
input_file = "unitary/test1.npz"   # The file exported from your GUI
output_file = "unitary/test1.npy"   # The new file you want to create

# 2. Load the JAX twin archive
data = np.load(input_file)

# 3. Extract the physical unitary matrix
# (Your GUI saves it under the 'unitary' key)
U_physical = data['unitary']

# 4. Save it as a pure .npy file
np.save(output_file, U_physical)

print(f"Successfully extracted a {U_physical.shape} unitary and saved to {output_file}!")