
import requests
import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_path = os.path.join(project_dir, 'Data', 'weather_monthly.csv')

print("Fetching weather data from Open-Meteo API:")
print("Location: Middlesbrough (54.57, -1.23)")

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 54.57,
    "longitude": -1.23,
    "start_date": "2023-04-01",
    "end_date": "2026-03-31",
    "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum"
}

response = requests.get(url, params=params)

if response.status_code != 200:
    print(f"ERROR: API returned status {response.status_code}")
    print(f"Response: {response.text[:500]}")
    exit()

data = response.json()
weather_daily = pd.DataFrame(data['daily'])
weather_daily['time'] = pd.to_datetime(weather_daily['time'])
weather_daily['Month'] = weather_daily['time'].dt.to_period('M').astype(str)

print(f"Daily records fetched: {len(weather_daily)}")
print(f"Date range: {weather_daily['time'].min().date()} to {weather_daily['time'].max().date()}")

# Aggregate to monthly
weather_monthly = weather_daily.groupby('Month').agg(
    temp_mean=('temperature_2m_mean', 'mean'),
    temp_max=('temperature_2m_max', 'mean'),
    temp_min=('temperature_2m_min', 'mean'),
    rainfall_total=('precipitation_sum', 'sum'),
    rainfall_days=('precipitation_sum', lambda x: (x > 0).sum())
).reset_index()

# Round for readability
weather_monthly['temp_mean'] = weather_monthly['temp_mean'].round(1)
weather_monthly['temp_max'] = weather_monthly['temp_max'].round(1)
weather_monthly['temp_min'] = weather_monthly['temp_min'].round(1)
weather_monthly['rainfall_total'] = weather_monthly['rainfall_total'].round(1)

#SAving to CSV
weather_monthly.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")
print(f"Months: {len(weather_monthly)}")
print(f"\nPreview:")
print(weather_monthly.to_string())
print("\nDone!")
