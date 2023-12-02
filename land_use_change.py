import xarray as xr
import numpy as np
import time
from tqdm import tqdm 

start = time.time()

# Path to NetCDF files
input_file = '/Users/jamesquessy/Desktop/Uni Work/Masters/Reasearch Project/Code/Power_Generation/land_use/land_use_uk_adjusted.nc'
output_file = '/Users/jamesquessy/Desktop/Uni Work/Masters/Reasearch Project/Code/Power_Generation/land_use/land_use_adjusted.nc'

# Open the dataset
ds = xr.open_dataset(input_file)

# IPCC classification mapping
ipcc_classes = {
    1: [10, 11, 12, 20, 30, 40],  # Agriculture
    2: [50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100, 160, 170],  # Forest
    3: [110, 130],  # Grassland
    4: [180],  # Wetland
    5: [190],  # Settlement
    6: [120, 121, 122, 140, 150, 151, 152, 153, 200, 201, 202],  # Other
    7: [210]  # Water
}

# Friction coefficient mapping
friction_coefficients = {
    1: 0.15,  # Agriculture -> Grasslands
    2: 0.25,  # Forest -> Heavily forested land
    3: 0.15,  # Grassland -> Grasslands
    4: 0.20,  # Wetland -> Tall crops, hedges, and shrubs
    5: 0.30,  # Settlement -> Small town with trees and shrubs
    6: 0.20,  # Other -> Tall crops, hedges, and shrubs
    7: 0.10   # Water -> Lakes, ocean, and smooth hard ground
}

# Adjust lccs_class values
new_lccs_class = ds['lccs_class'].values.copy()
for ipcc_class, lccs_values in ipcc_classes.items():
    for lccs_value in lccs_values:
        new_lccs_class[new_lccs_class == lccs_value] = ipcc_class

ds['lccs_class'].values = new_lccs_class

# Calculate friction coefficients
friction_coeff_array = np.vectorize(friction_coefficients.get)(new_lccs_class)

# Add new variable to dataset
ds['friction_coefficient'] = xr.DataArray(friction_coeff_array, dims=ds['lccs_class'].dims)

# Save updated dataset
ds.to_netcdf(output_file)

# Close dataset
ds.close()

end = time.time()
print(f'Elapsed time is {end - start} seconds')

""" 
After running this code you will have a new file which has changed the lcss_class variable to the new  subsections and 
a new variable has been made for the frictional coefficient. After running this we need to regrid teh data to the same
grid as the other files to do this we can use command from CDO: "cdo -L remapnn,UK_2020_acc.nc land_use_adjusted.nc remaped_land.nc". 
make sure to be in the same directory by running the command "cd land_use/".

"""