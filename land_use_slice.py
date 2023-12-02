import xarray as xr
import time 

start = time.time()

# Define the geographic boundaries of the UK
min_lon, max_lon = -10, 2  # UK longitudes approximately between -10°E and 2°E

# Path to the original NetCDF file
file_path = '/Users/jamesquessy/Desktop/Uni Work/Masters/Reasearch Project/Code/Power_Generation/land_use/land_use.nc'
output_path = '/Users/jamesquessy/Desktop/Uni Work/Masters/Reasearch Project/Code/Power_Generation/land_use/sliced_land_use_uk.nc'

# Open the NetCDF file
with xr.open_dataset(file_path) as ds:
    # Check the actual range of latitude and longitude in the dataset
    print("Actual latitude range:", ds.lat.min().values, "to", ds.lat.max().values)

    # Select the subset of the data for the UK
    sliced_ds1 = ds.sel(lon=slice(min_lon, max_lon))

    # Save the sliced data to a new file
    sliced_ds1.to_netcdf(output_path)

end = time.time()
print(f'elapsed time is {end - start} seconds')

"""
    This code is not perfect as you have to use it with other cdo commands and it does not perfectly slice the file but this code slices the lon value which 
    in turn makes it much easier to slice. It is then sliced using a cdo command in terminal to the size of the UK and then NaN values which are represented as 255
    are removed, leaving only data in the UK. I now need to re grid this to the same grid size as the other document. 
"""