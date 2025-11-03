import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

def eval_model(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print(f"{name:10s}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
    return model

print("α = 1.0 (default)")
ridge = eval_model("Ridge", Ridge(alpha=1.0))
lasso = eval_model("Lasso", Lasso(alpha=0.001, max_iter=10000))
