import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Load the data from the CSV file
csv_file_path = 'wind_farm_demand_satisfaction_all_years.csv'
model_locations = pd.read_csv(csv_file_path)

# Filter the data to remove rows where 'Daily_Power_Generation (kWh)' is 0, if necessary
# model_locations = model_locations[model_locations['Daily_Power_Generation (kWh)'] > 0]

# Create GeoDataFrame for model locations
# Ensure the column names match your CSV file's column headers
gdf_model = gpd.GeoDataFrame(model_locations, geometry=gpd.points_from_xy(model_locations.Best_Lon, model_locations.Best_Lat))

# Load Burgar Hill Wind Farm coordinates
burgar_hill_coords = [59.113944, -3.144306]

# Create GeoDataFrame for Burgar Hill Wind Farm
gdf_burgar_hill = gpd.GeoDataFrame({'name': ['Burgar Hill Wind Farm'], 'geometry': [Point(burgar_hill_coords[::-1])]})

# Plot for visual comparison
fig, ax = plt.subplots()
gdf_model.plot(ax=ax, color='blue', label='Model Locations')
gdf_burgar_hill.plot(ax=ax, color='red', label='Burgar Hill')
plt.legend()
plt.show()

