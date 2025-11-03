import pandas as pd
import numpy as np
from joblib import load
from datetime import datetime, timedelta

# Load the model
model=load("rf_model_timeaware.joblib")

# Load the latest data
df = pd.read_csv("restaurant_features.csv", parse_dates = ["datetime_hour"])
df = df.sort_values("datetime_hour")

# Get the most recent hour
latest = df.iloc[-1]
next_hour = latest["datetime_hour"] + timedelta(hours=1)

# Build the next-hour feature row
new_row = latest.copy()
new_row["datetime_hour"] = next_hour

# Update the lag values
new_row["lag_1h"] = latest["hourly_orders"]
new_row["lag_24h"] = df[df["datetime_hour"] == latest["datetime_hour"] - timedelta(hours=24)]["hourly_orders"].mean()

# Keep the same rolling means and weather for simplicity
feature_cols = [
    "hour_sin","hour_cos","is_weekend",
    "lag_1h","lag_24h",
    "roll_mean_6h","roll_mean_24h",
    "temp_max","temp_min","precip","rain_temp_effect"
]
X_next = new_row[feature_cols].to_frame().T

# Predict
predicted_orders = model.predict(X_next)[0]
print(f"Predicted orders for next hour ({next_hour}): {predicted_orders:.2f}")