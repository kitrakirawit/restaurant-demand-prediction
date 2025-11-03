import time
t0 = time.time()

import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump

# 1) Load
df = pd.read_csv("restaurant_features.csv")
feature_cols = [
    "hour_sin","hour_cos","is_weekend",
    "lag_1h","lag_24h",
    "roll_mean_6h","roll_mean_24h",
    "temp_max","temp_min","precip","rain_temp_effect"
]
X, y = df[feature_cols], df["hourly_orders"]

# 2) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3) Train (single model, with progress)
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    verbose=1  # progress updates
)
rf.fit(X_train, y_train)

# 4) Predict + evaluate
pred = rf.predict(X_test)
mae  = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2   = r2_score(y_test, pred)
print(f"RandomForest  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

# 5) Importances (from the FITTED model)
fi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importance (top 10):")
print(fi.head(10).to_string())

# 6) Save artifacts
with open("rf_metrics.txt", "w") as f:
    f.write(f"MAE={mae:.6f}\nRMSE={rmse:.6f}\nR2={r2:.6f}\n")

fi.to_csv("rf_feature_importance.csv")
dump(rf, "rf_model.joblib")
print("✅ Saved rf_model.joblib, rf_feature_importance.csv, and rf_metrics.txt")

print(f"Training time: {time.time() - t0:.1f}s")
