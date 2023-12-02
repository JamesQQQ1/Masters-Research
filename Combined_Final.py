import xarray as xr
import pandas as pd
import numpy as np
import os
import netCDF4 as nc
from netCDF4 import Dataset
from geopy.distance import great_circle

# Define base paths
base_directory = '/Users/jamesquessy/Desktop/Uni Work/Masters/Reasearch Project/Code/Power_Generation'
population_directory = os.path.join(base_directory, 'Population')
last_year_avg_directory = os.path.join(base_directory, 'last_year_avg')
merged_directory = os.path.join(base_directory, 'Merged/Merged_Files')

# Ensure the merged directory exists
os.makedirs(merged_directory, exist_ok=True)

# Define years and variables
years = ['2020', '2050', '2075', '2099']
variables = ['hurs', 'ps', 'sfcWind', 'tas']

# Static file paths
orography_file_path = os.path.join(last_year_avg_directory, 'orography_remap.nc')
land_area_file_path = os.path.join(last_year_avg_directory, 'land_area_remap.nc')
land_use_file_path = os.path.join(base_directory, 'land_use/remaped_land.nc')

# Constants for power generation
turbine_area = 2000  # Area swept by a single turbine's blades in square meters
power_coefficient = 0.35  # Power coefficient (Cp) of the turbine

# Function to calculate wind at 80m using friction coefficient
def calculate_wind_at_80m(wind_speed_10m, friction_coefficient):
    reference_height = 10
    target_height = 80
    wind_speed_80m = wind_speed_10m * (np.log(target_height / friction_coefficient) / np.log(reference_height / friction_coefficient))
    return wind_speed_80m

# Function to calculate saturation vapor pressure
def calculate_saturation_vapor_pressure(t):
    es = 6.1094 * np.exp((17.625 * t) / (t + 243.04))
    return es * 100

# Function to calculate vapor pressure
def calculate_vapor_pressure(t, rh):
    es = calculate_saturation_vapor_pressure(t)
    return (rh / 100.0) * es

# Function to calculate air density
def calculate_air_density(ps, tas, rh):
    Rd = 287.05
    Rv = 461.5
    temp_celsius = tas - 273.15
    e = calculate_vapor_pressure(temp_celsius, rh)
    Pd = ps - e
    return (Pd / (Rd * tas)) + (e / (Rv * tas))

# Function to calculate power generation for a single turbine
def calculate_power_generation(wind_80m, air_density, turbine_area, power_coefficient):
    # Power generation formula adjusted for a single turbine
    wind_power = 0.5 * air_density * turbine_area * (wind_80m ** 3) * power_coefficient
    return wind_power / 1000  # Convert to kW

# Function to merge datasets
def merge_datasets(year):
    # Open static datasets
    orography_ds = xr.open_dataset(orography_file_path)
    land_area_ds = xr.open_dataset(land_area_file_path)
    land_use_ds = xr.open_dataset(land_use_file_path)  # Open the land use dataset
    datasets = [orography_ds, land_area_ds, land_use_ds]  # Include land use dataset

    # Process variables and merge datasets
    for variable in variables:
        # File path for the variable
        file_path = os.path.join(last_year_avg_directory, f"{variable}_{year}_yearly_avg.nc")
        if os.path.exists(file_path):
            ds = xr.open_dataset(file_path)
            if 'height' in ds:
                ds = ds.drop_vars('height')
            datasets.append(ds)
        else:
            print(f"File not found: {file_path}")
            continue

    # Merge all datasets for the year
    merged_ds = xr.merge(datasets)

    # Calculate wind at 80m using the friction coefficient from the land use dataset
    if 'sfcWind' in merged_ds and 'friction_coefficient' in merged_ds:
        merged_ds['wind_80m'] = calculate_wind_at_80m(merged_ds['sfcWind'], merged_ds['friction_coefficient'])

    # Calculate air density
    if 'ps' in merged_ds and 'tas' in merged_ds and 'hurs' in merged_ds:
        merged_ds['air_density'] = calculate_air_density(merged_ds['ps'], merged_ds['tas'], merged_ds['hurs'])

    # Calculate power generation
    if 'wind_80m' in merged_ds and 'air_density' in merged_ds:
        # Call the updated function without the land fraction (sftlf)
        merged_ds['power_generation'] = calculate_power_generation(
            merged_ds['wind_80m'], 
            merged_ds['air_density'], 
            turbine_area, 
            power_coefficient
        )
    # Add attributes to the new variables
    if 'wind_80m' in merged_ds:
        merged_ds['wind_80m'].attrs = {
            'long_name': 'Wind Speed at 80m',
            'units': 'm/s',
            'description': 'Calculated wind speed at 80 meters above ground level.'
        }
    if 'air_density' in merged_ds:
        merged_ds['air_density'].attrs = {
            'long_name': 'Air Density',
            'units': 'kg/m^3',
            'description': 'Calculated air density at surface level.'
        }
    if 'power_generation' in merged_ds:
        merged_ds['power_generation'].attrs = {
            'long_name': 'Expected Wind Turbine Power Generation',
            'units': 'kW',
            'description': 'Calculated power output from a wind turbine based on wind speed at 80m and air density.'
        }

    # Save the merged dataset
    merged_file_path = os.path.join(merged_directory, f"Merged_{year}.nc")
    merged_ds.to_netcdf(merged_file_path)
    print(f"Merged file for {year} saved at {merged_file_path}")

    return merged_file_path

# Function to add electricity demand to NetCDF
def add_demand_to_netcdf(year, cities, base_nc_path, population_directory):
    for city in cities:
        csv_file_path = os.path.join(population_directory, year, f'{city}_pop_{year}.csv')
        if not os.path.exists(csv_file_path):
            print(f"CSV file not found for {city} in {year}")
            continue

        nc_file_path = os.path.join(base_nc_path, f'Merged_{year}.nc')
        nc_file = Dataset(nc_file_path, 'r+')  # Open the NetCDF file in read/write mode

        # Read the CSV data for the city
        csv_data = pd.read_csv(csv_file_path)

        # Construct the correct demand column name based on the year
        demand_column = f'demand_{year} (kWh)'  # Use format strings to insert the year
        if demand_column not in csv_data.columns:
            print(f"Column '{demand_column}' not found in {csv_file_path}")
            continue

        # Create a new variable for each city
        demand_var_name = f'Energy_Demand_{city}'
        demand_var = nc_file.createVariable(demand_var_name, 'f4', ('lat', 'lon'))
        demand_var.units = 'kWh'
        demand_var.long_name = f'Electricity Demand for {city} in {year}'
        demand_var[:] = np.nan  # Initialize with NaNs

        # Assign data to the variable
        for index, row in csv_data.iterrows():
            lat_idx = np.abs(nc_file.variables['lat'][:] - row['Latitude']).argmin()
            lon_idx = np.abs(nc_file.variables['lon'][:] - row['Longitude']).argmin()
            demand_value = float(row[demand_column]) if np.isfinite(row[demand_column]) else np.nan
            demand_var[lat_idx, lon_idx] = demand_value

        # Save and close the NetCDF file
        nc_file.close()

# Define the variables to be dropped
variables_to_drop = ["air_density", "change_count", 'friction_coefficient', 'hurs',
                     'current_pixel_state', 'observation_count', 'orog', 'processed_flag',
                     'ps', 'sfcWind', 'sftlf', 'tas', 'time', 'time_bnds', 'wind_80m']

# Create a new directory for the final files within the current working directory
final_files_directory = os.path.join(os.getcwd(), "final_files")
os.makedirs(final_files_directory, exist_ok=True)

for year in years:
    # Step 1: Merge datasets
    merged_file_path = merge_datasets(year)
    
    # Step 2: Process and drop variables
    essential_var_file_path = os.path.join(merged_directory, f"essential_var_{year}.nc")
    if os.path.exists(merged_file_path):
        ds = xr.open_dataset(merged_file_path)
        ds = ds.drop_vars(variables_to_drop)
        if not os.path.exists(essential_var_file_path):
            ds.to_netcdf(essential_var_file_path)
            print(f"Essential variables saved for {year}")
        else:
            print(f"Essential variables file already exists for {year}")
        ds.close()
    else:
        print(f"Failed to process file for {year}")

    # Step 3: Apply masks
    if os.path.exists(essential_var_file_path):
        dataset = Dataset(essential_var_file_path, 'r+')
        lccs_class = dataset.variables['lccs_class'][:]
        power_generation = dataset.variables['power_generation'][:]
        urban_mask = lccs_class == 5
        water_mask = lccs_class == 2
        exclusion_mask = np.logical_or(urban_mask, water_mask)
        power_generation_masked = np.ma.array(power_generation, mask=exclusion_mask)
        dataset.variables['power_generation'][:] = power_generation_masked
        dataset.sync()
        dataset.close()
        print(f"Masking applied and saved for {year}")
    else:
        print(f"Failed to apply masks for {year}")

    # Step 4: Replace NaN values with 0 and save in the new directory
    final_file_path = os.path.join(final_files_directory, f"final_file_{year}.nc")
    if os.path.exists(essential_var_file_path):
        ds = xr.open_dataset(essential_var_file_path)
        for var in ds.variables:
            if ds[var].dtype.kind in 'f':
                ds[var] = ds[var].fillna(0)
        if not os.path.exists(final_file_path):
            ds.to_netcdf(final_file_path)
            print(f"All NaN Values removed and saved in 'final_files' directory for {year}")
        else:
            print(f"Final file already exists in 'final_files' directory for {year}")
        ds.close()
    else:
        print(f"Failed to replace NaN values for {year}")
        

# Constants
POWER_LOSS_PER_1000KM = 0.0035  # 3.5% power loss every 1000 km
DAYS_PER_YEAR = 365  # Total days in a year

# Function to calculate power loss over distance
def calculate_power_loss(power, distance):
    distance_km = distance / 1000
    loss_fraction = 1 - (POWER_LOSS_PER_1000KM * (distance_km // 1000))
    return power * loss_fraction

# Initialize an empty DataFrame to store results for all years
all_years_best_locations = pd.DataFrame()

# Process data for each year
years = ['2020', '2050', '2075', '2099']
for year in years:
    # Load netCDF data for wind power generation for the current year
    file_path = f'final_files/final_file_{year}.nc'
    dataset = nc.Dataset(file_path)

    # Extracting wind power data
    lon = dataset.variables['lon'][:]
    lat = dataset.variables['lat'][:]
    power_generation = dataset.variables['power_generation'][:,:,0].filled(np.nan)

    # Load city energy demand data from CSV file for the current year
    energy_demand_df = pd.read_csv(f'Population/city_power_demand_projection_{year}.csv')

    # DataFrame to store results for the current year
    best_locations = pd.DataFrame()

    # Iterate over each city in the CSV file
    for index, row in energy_demand_df.iterrows():
        city_name = row['City']
        city_coords = (row['Latitude'], row['Longitude'])
        city_energy_demand_annual = row['Energy Demand (kWh)']

        best_power = 0
        best_location = None
        best_distance = None

        # Iterate over each grid point to find the best location
        for i in range(len(lat)):
            for j in range(len(lon)):
                daily_power_generation = power_generation[i, j]
                if daily_power_generation > 0:  # Exclude unsuitable locations
                    wind_farm_coords = (lat[i], lon[j])
                    distance = great_circle(wind_farm_coords, city_coords).kilometers
                    adjusted_daily_power = calculate_power_loss(daily_power_generation, distance)
                    if adjusted_daily_power > best_power:
                        best_power = adjusted_daily_power
                        best_location = wind_farm_coords
                        best_distance = distance

        # Calculate the annual energy production for the best location
        annual_energy_production = best_power * DAYS_PER_YEAR

        # Calculate percentage of city demand that could be satisfied
        demand_satisfaction = (annual_energy_production / city_energy_demand_annual) if city_energy_demand_annual else 0

        # Append results to the DataFrame for the current year
        new_row = {
            'Year': year,
            'City': city_name,
            'Best_Lat': best_location[0] if best_location else np.nan,
            'Best_Lon': best_location[1] if best_location else np.nan,
            'Distance_to_City (km)': best_distance if best_distance else np.nan,
            'Adjusted_Daily_Power (kWh)': best_power,
            'Annual_Energy_Production (kWh)': annual_energy_production,
            'City_Energy_Demand (kWh)': city_energy_demand_annual,
            'Demand_Satisfaction (%)': demand_satisfaction
        }
        best_locations = pd.concat([best_locations, pd.DataFrame([new_row])], ignore_index=True)

    # Append the results of the current year to the aggregate DataFrame
    all_years_best_locations = pd.concat([all_years_best_locations, best_locations], ignore_index=True)

    # Close the dataset for the current year
    dataset.close()

    # Print completion message for the current year
    print(f"The analysis for {year} has been completed.")

# Round all numeric values in the DataFrame to one decimal place
all_years_best_locations = all_years_best_locations.round( 10)

# Save the aggregated results to a single CSV file
all_years_best_locations.to_csv('wind_farm_demand_satisfaction_all_years.csv', index=False)
print("All years processed successfully. Results saved to 'wind_farm_demand_satisfaction_all_years.csv'")