import pandas as pd
from datetime import datetime

# Loading the data
daily = pd.read_csv("restaurant_with_weather.csv")
pattern= pd.read_csv("hourly_pattern.csv")

# Making sure the date is parsed as date
daily["visit_date"] = pd.to_datetime(daily["visit_date"])

# Cross join
hourly = daily.merge(pattern, how="cross")

# build timestamp
hourly["datetime_hour"] = hourly["visit_date"] + pd.to_timedelta(hourly["hour"], unit ="h")

# Compute hourly orders
hourly["hourly_orders_float"] = hourly["visitors"] * hourly["share"]

# Rounding 
hourly["hourly_orders"] = hourly["hourly_orders_float"]

# Check reconstruction integrity by day
# Group back by store/date to see sums ≈ original visitors
check = (hourly
         .groupby(["air_store_id","visit_date"], as_index=False)["hourly_orders"]
         .sum())
mismatch = (check.merge(daily[["air_store_id","visit_date","visitors"]],
                        on=["air_store_id","visit_date"])
                 .assign(diff=lambda d: (d["hourly_orders"] - d["visitors"]).abs()))
print("Mean reconstruction error:", mismatch["diff"].mean())

# Keeping tidy columns and saving
cols = [
    "air_store_id", "datetime_hour", "hour", "hourly_orders",
    "visit_date", "visitors", "day_of_week", "holiday_flg",
    "air_genre_name", "air_area_name",
    "temp_max", "temp_min", "precip"
]
hourly = hourly[cols].sort_values(["air_store_id","datetime_hour"])

hourly.to_csv("restaurant_hourly.csv", index=False)
print("Wrote restaurant_hourly.csv with", hourly.shape[0], "rows")
