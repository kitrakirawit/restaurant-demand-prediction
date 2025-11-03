import requests
import pandas as pd

# Defining the City coordinates(for tokyo)
lat, long = 35.68, 139.76

# Defining data range
start_date, end_date = "2016-01-01", "2017-04-22"

# Building API URL
url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={lat}&longitude={long}"
    f"&start_date={start_date}&end_date={end_date}"
    "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
    "&timezone=auto"
    )

# Fetching Data
response= requests.get(url)
data = response.json()

weather = pd.DataFrame({
    "date": data["daily"]["time"],
    "temp_max": data["daily"]["temperature_2m_max"],
    "temp_min": data["daily"]["temperature_2m_min"],
    "precip": data["daily"]["precipitation_sum"]
})

weather.to_csv("tokyo_weather.csv",index=False)
print("Saved tokyo weather csv file with", weather.shape[0],"rows")