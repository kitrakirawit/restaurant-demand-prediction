import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("restaurant_features.csv")

feature_cols = [
    "hour_sin","hour_cos","is_weekend",
    "lag_1h","lag_24h",
    "roll_mean_6h","roll_mean_24h",
    "temp_max","temp_min","precip","rain_temp_effect"
]
X, y = df[feature_cols], df["hourly_orders"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression().fit(X_train, y_train)

coef = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": model.coef_
}).sort_values("coefficient", key=abs, ascending=False)

print(coef.to_string(index=False))
print("\nIntercept:", model.intercept_)
