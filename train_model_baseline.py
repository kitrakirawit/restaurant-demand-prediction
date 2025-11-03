import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np 
import matplotlib.pyplot as plt

# Lets load the data first
df = pd.read_csv("restaurant_features.csv")

# Selecting the features (as inputs) and target(output)
feature_cols = [
    "hour_sin", "hour_cos", "is_weekend",
    "lag_1h", "lag_24h",
    "roll_mean_6h", "roll_mean_24h",
    "temp_max", "temp_min", "precip", "rain_temp_effect"
]
X = df[feature_cols]
y = df["hourly_orders"]

# Split data into training and testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Trained")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")

# Save predictions
results = pd.DataFrame({"actual": y_test, "predicted": y_pred})
results.to_csv("predictions_baseline.csv", index=False)
print("Saved predictions_baseline.csv")

# Scatter plot
plt.scatter(y_test[:200], y_pred[:200], alpha=0.6)
plt.xlabel("Actual Orders")
plt.ylabel("Predicted Orders")
plt.title("Predicted vs Actual (Sample 200)")
plt.plot([0, max(y_test[:200])], [0, max(y_test[:200])], color="red")
plt.show()