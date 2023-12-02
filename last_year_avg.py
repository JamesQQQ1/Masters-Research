""" 
This code takes all of the files and slices them so we only get teh last year and then averages then so we 
get teh daily average over the years. This only needs to be run as a back up incase the files are lost or 
incase the files are damaged.
"""

import os
import xarray as xr

def extract_last_year(file_path, new_folder, var_name, year):
    """
    Extract the yearly average data from a NetCDF file and save to a new file.

    Parameters:
    - file_path: path to the original NetCDF file
    - new_folder: folder to save the new NetCDF files with the yearly average data
    - var_name: variable name like 'tas', 'hurs', etc.
    - year: year like '2020', '2050', etc.
    """

    # Create the folder if it doesn't exist
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    # Path for the new file
    new_file_path = os.path.join(new_folder, f"{var_name}_{year}_yearly_avg.nc")

    # Load the dataset
    data = xr.open_dataset(file_path, engine='netcdf4')

    # Select the data for the specific year
    yearly_data = data.sel(time=data.time.dt.year == int(year))

    # Compute the average for the year
    avg_data = yearly_data[var_name].mean(dim='time')

    # Create a new dataset with the average data
    avg_dataset = xr.Dataset({var_name: avg_data})
    avg_dataset.attrs = data.attrs

    # Save the average data to the new file path
    avg_dataset.to_netcdf(new_file_path)
    print(f"Saved {new_file_path}")

# Define the base path where the original files are located
original_file_path = '/Users/jamesquessy/Desktop/Uni Work/Masters/Reasearch Project/Code/Power_Generation /NetCDF_Files'

# Define the folder where the new files will be saved
new_folder = '/Users/jamesquessy/Desktop/Uni Work/Masters/Reasearch Project/Code/Power_Generation/last_year_avg'

# List of years and variables to process
years = ['2020', '2050', '2075', '2099']
variables = ['tas', 'hurs', 'sfcWind', 'ps']

# Loop through each year and variable, process the file and save the average
for year in years:
    for var_name in variables:
        # Construct the file path for the specific .nc file you want to process
        file_name = f"{var_name}_{year}_remap.nc"  # Adjusted to match your file naming convention
        file_path = os.path.join(original_file_path, file_name)

        # Print the full path for debugging
        print(f"Looking for file at: {file_path}")

        # Check if the file exists before trying to open it
        if os.path.isfile(file_path):
            extract_last_year(file_path, new_folder, var_name, year)
        else:
            print(f"File not found: {file_path}")
