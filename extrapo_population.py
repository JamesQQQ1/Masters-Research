import pandas as pd

# Starting energy demand
starting_demand = 5130

# Forecasted demand changes
forecasted_changes = {
    '2020': 0.0,        # 0% change
    '2050': 10.904668, # -10.904668% improvement
    '2075': 20.784495, # -20.784495% improvement
    '2099': 30.269128  # -30.269128% improvement
}

# Projected efficiency improvements
efficiency_improvements = {
    '2020': 0.0,    # 0% improvement
    '2050': 0.20,   # 20% improvement
    '2075': 0.30,   # 30% improvement
    '2099': 0.40    # 40% improvement
}

# Calculating adjusted demand
adjusted_demands = {}
for year, change in forecasted_changes.items():
    forecasted_demand = starting_demand * (1 + (change / 100))
    efficiency_improvement = efficiency_improvements[year]
    adjusted_demand = forecasted_demand * (1 - efficiency_improvement)
    adjusted_demands[year] = adjusted_demand

print (adjusted_demands)

per_capita_demand = {
    '2020': adjusted_demands, 
    '2050': adjusted_demands,  
    '2075': adjusted_demands, 
    '2099': adjusted_demands  
}

# Define the total population for the given years0.
total_population_years = {
    2020: 67081234,
    2050: 73162612,
    2075: 74948129,
    2099: 76015656
}

# Define the population and coordinates for cities in 2020.
city_data = {
    'London': {'Population': 9304000.00, 'Latitude': 51.5085, 'Longitude': -0.1257},
    'Manchester': {'Population': 2730000.00, 'Latitude': 53.481, 'Longitude': -2.2374},
    'Birmingham': {'Population': 2607000.00, 'Latitude': 52.4814, 'Longitude': -1.8998},
    'Leeds': {'Population': 1889000.00, 'Latitude': 53.7984, 'Longitude': -1.7649},
    'Glasgow': {'Population': 1673000.00, 'Latitude': 55.8652, 'Longitude': -4.2576},
    'Southampton': {'Population': 928000.00, 'Latitude': 50.9117, 'Longitude': -1.4036},
    'Liverpool': {'Population': 902000.00, 'Latitude': 53.4106, 'Longitude': -2.9779},
    'Newcastle upon Tyne': {'Population': 809000.00, 'Latitude': 54.9733, 'Longitude': -1.614},
    'Nottingham': {'Population': 788000.00, 'Latitude': 52.9536, 'Longitude': -1.1505},
    'Sheffield': {'Population': 730000.00, 'Latitude': 53.383, 'Longitude': -1.4659},
    'Bristol': {'Population': 686000.00, 'Latitude': 51.4552, 'Longitude': -2.5966},
    'Belfast': {'Population': 631000.00, 'Latitude': 54.5947, 'Longitude': -5.9298},
    'Brighton': {'Population': 607000.00, 'Latitude': 50.8284, 'Longitude': -0.1395},
    'Leicester': {'Population': 552000.00, 'Latitude': 52.6386, 'Longitude': -1.1317},
    'Edinburgh': {'Population': 537000.00, 'Latitude': 55.9521, 'Longitude': -3.1965},
    'Bournemouth': {'Population': 506000.00, 'Latitude': 50.7205, 'Longitude': -1.8795},
    'Cardiff': {'Population': 478000.00, 'Latitude': 51.48, 'Longitude': -3.18},
    'Coventry': {'Population': 426000.00, 'Latitude': 52.4066, 'Longitude': -1.5122},
    'Middlesbrough': {'Population': 387000.00, 'Latitude': 54.5762, 'Longitude': -1.2348},
    'Stoke-on-Trent': {'Population': 386000.00, 'Latitude': 53.0042, 'Longitude': -2.1854},
    'Reading': {'Population': 342000.00, 'Latitude': 51.4112, 'Longitude': -0.8356},
    'Sunderland': {'Population': 341000.00, 'Latitude': 54.9119, 'Longitude': -1.3833},
    'Birkenhead': {'Population': 329000.00, 'Latitude': 53.3934, 'Longitude': -3.0148},
    'Preston': {'Population': 328000.00, 'Latitude': 53.761, 'Longitude': -2.7024},
    'Kingston upon Hull': {'Population': 321000.00, 'Latitude': 53.7483, 'Longitude': -0.334},
    'Newport': {'Population': 316000.00, 'Latitude': 51.5847, 'Longitude': -2.9979},
    'Southend-On-Sea': {'Population': 312000.00, 'Latitude': 51.5378, 'Longitude': 0.7143},
    'Swansea': {'Population': 311000.00, 'Latitude': 51.6208, 'Longitude': -3.9432}
}


# Calculate population growth factor from the total population projection
population_growth_factors = {2020: 1}  # Include 2020 with a growth factor of 1
for year, population in total_population_years.items():
    if year != 2020:  # Skip the base year
        population_growth_factors[year] = population / total_population_years[2020]

# Project the population for each city based on the growth factor
projected_city_data = {}
for city, data in city_data.items():
    projected_city_data[city] = {}
    base_population = data['Population']
    projected_city_data[city][2020] = {
        'Population': base_population, 
        'Latitude': data['Latitude'], 
        'Longitude': data['Longitude']
    }
    for year, growth_factor in population_growth_factors.items():
        if year != 2020:
            projected_population = base_population * growth_factor
            projected_city_data[city][year] = {
                'Population': projected_population,
                'Latitude': data['Latitude'],
                'Longitude': data['Longitude']
            }

# For each year, create a CSV file with the projected demand for each city.
for year in population_growth_factors.keys():
    # Initialize a list to hold the projected demand for each city.
    city_energy_demand = []

    # Use the respective per capita demand for each year
    current_per_capita_demand = per_capita_demand[str(year)][str(year)]

    for city, data in projected_city_data.items():
        projected_info = data[year]
        projected_population = projected_info['Population']
        energy_demand = projected_population * current_per_capita_demand
        city_energy_demand.append({
            'City': city,
            'Year': year,
            'Projected Population': round(projected_population, 2),
            'Energy Demand (kWh)': round(energy_demand, 2),
            'Latitude': projected_info['Latitude'],
            'Longitude': projected_info['Longitude']
        })

    # Convert the list to a DataFrame.
    energy_demand_df = pd.DataFrame(city_energy_demand)

    # Save the DataFrame to a CSV file named after the year.
    csv_file_name = f'city_power_demand_projection_{year}.csv'
    energy_demand_df.to_csv(csv_file_name, index=False, float_format='%.2f')

    print(f"Saved projected energy demand for {year} to {csv_file_name}")