import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import matplotlib.pyplot as plt

# Prepare the data
data = {
    'Year': list(range(1950, 2019)),
    'TotalEnergyConsumption': [
        136442, 141130, 141300, 143368, 149065, 152003, 155518, 152411, 152537, 150596, 163175, 163769, 168733, 176040, 177992, 184289, 185463, 185528, 191614, 197482, 210095, 207167, 211117, 220507, 210391, 202192, 205635, 210872, 211798, 222011, 204492, 198360, 196069, 196764, 196402, 205657, 210010, 211038, 213098, 211433, 213687, 219505, 216815, 220564, 217491, 218421, 229988, 226814, 230743, 231328, 234807, 236855, 229605, 231867, 233633, 236290, 233073, 227492, 225573, 211634, 219546, 203667, 208120, 206786, 194042, 196487, 193493, 186393, 191423
    ]
}

# Convert the dictionary into a pandas DataFrame
df = pd.DataFrame(data)

# Convert 'Year' to DateTime format, set it as the index of the DataFrame, and specify the frequency
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df.set_index('Year', inplace=True)
df.index = pd.DatetimeIndex(df.index.values, freq=df.index.inferred_freq)

# ARIMA model
# Automatically determine the best ARIMA model parameters
arima_model = auto_arima(df, trace=True, error_action='ignore', suppress_warnings=True)

# Create and fit the ARIMA model with the determined order
model = ARIMA(df, order=arima_model.order)
model_fit = model.fit()

# Forecasting
# Define the number of periods (years) to forecast
n_periods = 81
# Generate forecast results
forecast_results = model_fit.get_forecast(steps=n_periods)
# Extract the forecasted mean values
forecast = forecast_results.predicted_mean
# Extract the confidence intervals for the forecasts
conf_int = forecast_results.conf_int()

# Extract specific forecasted values for the years 2020, 2050, 2075, and 2099
forecast_2020 = forecast['2020']
forecast_2050 = forecast['2050']
forecast_2075 = forecast['2075']
forecast_2099 = forecast['2099']

# Print the extracted forecasted values
print(f"Forecast for 2020: {forecast_2020}")
print(f"Forecast for 2050: {forecast_2050}")
print(f"Forecast for 2075: {forecast_2075}")
print(f"Forecast for 2099: {forecast_2099}")

# Check and plot forecast
# Ensure that the forecast does not contain null values
if not forecast.isnull().any():
    # Create a range of future years for plotting
    future_years = pd.date_range(start=df.index[-1] + pd.DateOffset(years=1), periods=n_periods, freq='Y')
    # Create a DataFrame for the forecast values and confidence intervals
    forecast_df = pd.DataFrame({'Forecast': forecast.values}, index=future_years)
    forecast_df['Lower_CI'] = conf_int.iloc[:, 0]
    forecast_df['Upper_CI'] = conf_int.iloc[:, 1]

    # Plotting
    plt.figure(figsize=(12,6))
    plt.plot(df, label='Historical')  # Plot historical data
    plt.plot(forecast_df['Forecast'], label='Forecast', color='red')  # Plot forecasted data
    plt.fill_between(future_years, forecast_df['Lower_CI'], forecast_df['Upper_CI'], color='pink', alpha=0.3)  # Fill confidence interval
    plt.title('Energy Consumption Forecast')
    plt.xlabel('Year')
    plt.ylabel('Total Energy Consumption (Thousand tonnes of oil equivalent)')
    plt.legend()
    plt.show()

else:
    # If forecast generation fails, print an error message
    print("Forecast generation failed. Please check the model fitting and data.")

# Forecasted energy demand data
data = {
    'Year': [2020, 2050, 2075, 2099],
    'Forecasted Demand': [194598.996549, 173378.62258, 154152.57854, 135695.576252]
}

# Create DataFrame
df = pd.DataFrame(data)
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df.set_index('Year', inplace=True)

# Calculate percentage change relative to the year 2020
baseline_2020 = df.loc['2020-01-01', 'Forecasted Demand']
df['Percentage Change from 2020'] = ((df['Forecasted Demand'] - baseline_2020) / baseline_2020) * 100

# Display the DataFrame with percentage changes
print(df)