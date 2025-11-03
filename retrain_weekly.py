import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
import numpy as np

# Load full updated data
df = pd.read_csv("restaurant_features.csv")
feature_cols = [
    "hour_sin","hour_cos","is_weekend",
    "lag_1h","lag_24h",
    "roll_mean_6h","roll_mean_24h",
    "temp_max","temp_min","precip","rain_temp_effect"
]
X, y = df[feature_cols], df["hourly_orders"]

# Retrain model
rf = RandomForestRegressor(
    n_estimators=200,
    n_jobs=-1,
    random_state=42
)
rf.fit(X, y)

# Save new version
dump(rf, "rf_model_timeaware.joblib")
print("Model retrained successfully and saved.")
