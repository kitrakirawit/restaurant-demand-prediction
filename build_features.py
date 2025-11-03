import pandas as pd
import numpy as np

df = pd.read_csv("restaurant_hourly.csv", parse_dates=["datetime_hour","visit_date"])
df = df.sort_values(["air_store_id", "datetime_hour"]).reset_index(drop=True)

# We will use sin/cos to loop through 0 to 23 naturally since natural numbers wouldn't get the point of the loop
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# Flagging the weekend
df["is_weekend"] = df["day_of_week"].isin(["Saturday","Sunday"]).astype(int)

# Lag feature: Look back at previous demand
df["lag_1h"]  = df.groupby("air_store_id")["hourly_orders"].shift(1)
df["lag_24h"] = df.groupby("air_store_id")["hourly_orders"].shift(24)

# Rolling averages 
# 6h - Short term
# 24h - Daily trend
df["roll_mean_6h"] = (
    df.groupby("air_store_id")["hourly_orders"]
      .rolling(6, min_periods=1).mean()
      .reset_index(level=0, drop=True)
)
df["roll_mean_24h"] = (
    df.groupby("air_store_id")["hourly_orders"]
      .rolling(24, min_periods=1).mean()
      .reset_index(level=0, drop=True)
)

# Teach the model that weather can affect orders in different temps
df["rain_temp_effect"] = df["precip"] * df["temp_max"]

# Since the Lags and rolling means cannot exist for the first few hours we drop it
df = df.dropna(subset=["lag_1h", "lag_24h"]).reset_index(drop=True)

df.to_csv("retaurant_features.csv",index=False)
print("Saved restaurant_features.csv with", df.shape[0],"rows and", df.shape[1], "columns")