import tidy3d as td
import matplotlib.pyplot as plt

# 1. Define base variables FIRST
num_trench, l_trench, gap_y = 4, 40.0, 1.0
total_array_length = num_trench * l_trench + (num_trench - 1) * gap_y

w_sim = 24
y_sim_size = total_array_length + 4
z_sim_min = -2  # Rough bottom of wafer
z_sim_max = 5   # Rough top of cladding

# 3. Re-calculate the slice coordinates
h_box = 2
h_core = 0.22
z_core = h_box + h_core / 2

y_start = -total_array_length / 2 + l_trench / 2
y_slice = y_start + l_trench / 2 

# Your downloaded file
file_path = "/home/foosa/Documents/tidy3d/heat_sim_check_mesh (hec-6c1e139f-430d-48ff-914c-8224347f60c9_v1).hdf5"

print("1. Loading heavy 3D data into local RAM...")
heat_sim_data = td.HeatChargeSimulationData.from_file(file_path)

print("2. Data loaded! Slicing 3D mesh into 2D...")
fig, ax = plt.subplots(1, 2, figsize=(14, 8)) 

# --- PLOT 1: Y-Plane Cross Section (XZ view) ---
temp_data_slice = heat_sim_data["temperature"].temperature.sel(y=y_slice, method="nearest")
temp_data_slice.plot(
    ax=ax[0], 
    cmap='inferno', 
    cbar_kwargs={'label': 'Temperature (K)', 'shrink': 0.5}
)
ax[0].set_title(f"Cross-Section (y={y_slice:.2f})")
# Force centering on X and Z
ax[0].set_xlim([-w_sim / 2, w_sim / 2])
ax[0].set_ylim([z_sim_min, z_sim_max])
ax[0].set_aspect('equal') 


# --- PLOT 2: Top-Down View (XY view) ---
print("3. Slicing Top-down view...")
temp_data_xy = heat_sim_data["temperature"].temperature.sel(z=z_core, method="nearest")
temp_data_xy.plot(
    ax=ax[1], 
    cmap='inferno', 
    cbar_kwargs={'label': 'Temperature (K)', 'shrink': 0.8}
)
ax[1].set_title(f"Top-Down Field (z={z_core:.2f})")
# Force centering on X and Y
ax[1].set_xlim([-w_sim / 2, w_sim / 2])
ax[1].set_ylim([-y_sim_size / 2, y_sim_size / 2])
ax[1].set_aspect('equal') 

plt.tight_layout()
print("4. Rendering plot...")
plt.show()