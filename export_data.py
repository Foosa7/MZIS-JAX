import tidy3d as td

# Update the file path to match your actual file location
file_path = "/home/foosa/Documents/tidy3d/heat_sim_check_mesh (hec-6c1e139f-430d-48ff-914c-8224347f60c9_v1).hdf5"

print("Loading HeatCharge data...")
# FIXED: Changed td.SimulationData to td.HeatChargeSimulationData
heat_sim_data = td.HeatChargeSimulationData.from_file(file_path)

print("Data loaded successfully!")

# Now tell the temperature monitor to export to VTK
print("Converting to VTK...")
heat_sim_data["temperature"].to_vtk(file_path="my_3d_heatsim_temperature")

print("Done! You can now open the .vtu file in ParaView.")