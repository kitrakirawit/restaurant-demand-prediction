import time
t0= time.time()

import pandas as pd , numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump
import matplotlib.pyplot as plt

# 1️⃣ Load data and sort by time
df = pd.read_csv("restaurant_features.csv", parse_dates=["datetime_hour"])
df = df.sort_values("datetime_hour")

feature_cols = [
    "hour_sin","hour_cos","is_weekend",
    "lag_1h","lag_24h",
    "roll_mean_6h","roll_mean_24h",
    "temp_max","temp_min","precip","rain_temp_effect"
]

# 2️⃣ Split by time (train on first 80%, test on last 20%)
cutoff = int(len(df) * 0.8)
train, test = df.iloc[:cutoff], df.iloc[cutoff:]

X_train, y_train = train[feature_cols], train["hourly_orders"]
X_test, y_test   = test[feature_cols], test["hourly_orders"]

# 3️⃣ Train Random Forest
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    verbose=1
)
rf.fit(X_train, y_train)

# 4️⃣ Evaluate
pred = rf.predict(X_test)
mae  = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2   = r2_score(y_test, pred)

print(f"\nTime-aware Validation:")
print(f"MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
print(f"Training time: {time.time()-t0:.1f}s")

# 5️⃣ Save metrics
with open("rf_metrics_timeaware.txt", "w") as f:
    f.write(f"MAE={mae:.6f}\nRMSE={rmse:.6f}\nR2={r2:.6f}\n")

dump(rf, "rf_model_timeaware.joblib")

# 6️⃣ Visualize: Predicted vs Actual (sample)
plt.figure(figsize=(8,6))
plt.scatter(y_test[:500], pred[:500], alpha=0.5)
plt.xlabel("Actual Orders")
plt.ylabel("Predicted Orders")
plt.title("Predicted vs Actual (Time-Aware Split)")
plt.plot([0, max(y_test[:500])], [0, max(y_test[:500])], color="red")
plt.show()

# 7️⃣ Visualize: Feature Importance
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
plt.figure(figsize=(8,6))
importances.plot(kind="barh")
plt.title("Feature Importance (Time-Aware RF)")
plt.tight_layout()
plt.show()