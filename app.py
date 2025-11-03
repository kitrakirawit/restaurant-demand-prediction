# app.py — diagnostic-first version
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import timedelta
from joblib import load
import streamlit as st

st.set_page_config(page_title="Restaurant Demand Prediction", layout="centered")

st.title("Short-Term Restaurant Demand Prediction")
st.caption("If you can read this, the app rendered successfully.")

# -------------------- Quick environment diagnostics --------------------
with st.expander("Diagnostics (click to expand)"):
    cwd = os.getcwd()
    st.write("**Working directory**:", cwd)
    st.write("**Files at repo root:**")
    try:
        st.write(sorted(os.listdir("."))[:100])
    except Exception as e:
        st.exception(e)

    # Show presence of key paths/files
    candidates = [
        "restaurant_features_demo.csv",
        "data/processed/restaurant_features_demo.csv",
        "data/processed/restaurant_features.csv",
        "restaurant_features.csv",
        "model_registry/rf_model_timeaware.joblib",
        "rf_model_timeaware.joblib",
    ]
    existence = {p: Path(p).exists() for p in candidates}
    st.write("**Key paths exist?**")
    st.json(existence)

# -------------------- Config --------------------
FEATURE_COLS = [
    "hour_sin", "hour_cos", "is_weekend",
    "lag_1h", "lag_24h",
    "roll_mean_6h", "roll_mean_24h",
    "temp_max", "temp_min", "precip", "rain_temp_effect"
]

DATA_CANDIDATES = [
    "restaurant_features_demo.csv",
    "data/processed/restaurant_features_demo.csv",
    "data/processed/restaurant_features.csv",
    "restaurant_features.csv",
]

MODEL_CANDIDATES = [
    "model_registry/rf_model_timeaware.joblib",
    "rf_model_timeaware.joblib",
    "model_registry/rf_model.joblib",
    "rf_model.joblib",
]

# -------------------- Loaders --------------------
@st.cache_data(show_spinner=True)
def load_small_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime_hour"])
    df = df.sort_values(["air_store_id", "datetime_hour"])
    if "hour" not in df.columns:
        df["hour"] = df["datetime_hour"].dt.hour
    return df

@st.cache_data(show_spinner=True)
def load_full_chunked(path: str, max_rows: int = 400_000) -> pd.DataFrame:
    """Load a subset of the large CSV to avoid memory blowups on Streamlit Cloud."""
    chunks = pd.read_csv(path, parse_dates=["datetime_hour"], chunksize=100_000)
    acc = []
    total = 0
    for ch in chunks:
        acc.append(ch)
        total += len(ch)
        if total >= max_rows:
            break
    df = pd.concat(acc, ignore_index=True)
    df = df.sort_values(["air_store_id", "datetime_hour"])
    if "hour" not in df.columns:
        df["hour"] = df["datetime_hour"].dt.hour
    return df

@st.cache_resource(show_spinner=True)
def load_model() -> object:
    last_err = None
    for p in MODEL_CANDIDATES:
        try:
            return load(p)
        except Exception as e:
            last_err = e
            continue
    raise FileNotFoundError(f"Model not found. Tried: {MODEL_CANDIDATES}. Last error: {last_err}")

def build_next_row(store_df: pd.DataFrame) -> pd.Series:
    last = store_df.iloc[-1].copy()
    next_dt = last["datetime_hour"] + timedelta(hours=1)
    new = last.copy()
    new["datetime_hour"] = next_dt

    h = int(next_dt.hour)
    new["hour_sin"] = np.sin(2 * np.pi * h / 24)
    new["hour_cos"] = np.cos(2 * np.pi * h / 24)
    new["is_weekend"] = int(next_dt.day_name() in ["Saturday", "Sunday"])

    if "hourly_orders" in store_df.columns:
        new["lag_1h"] = float(last["hourly_orders"])
    else:
        new["lag_1h"] = np.nan

    prev24 = store_df.loc[store_df["datetime_hour"] == (next_dt - timedelta(hours=24)), "hourly_orders"]
    new["lag_24h"] = float(prev24.iloc[0]) if not prev24.empty else np.nan

    return new

# -------------------- UI Controls --------------------
st.subheader("Load data")
load_mode = st.radio(
    "Choose how to load data:",
    ["Fast demo (if present)", "Chunked full file (safe subset)", "Direct full file (may stall)"],
    index=0
)

# Resolve a candidate path
data_path = None
for p in DATA_CANDIDATES:
    if Path(p).exists():
        data_path = p
        break

if data_path is None:
    st.error(
        "No data file found. Please add one of these to the repo: "
        f"{DATA_CANDIDATES}"
    )
    st.stop()

st.write("**Selected data path:**", data_path)

try:
    if load_mode == "Fast demo (if present)":
        if "demo" not in data_path:
            st.warning("Demo file not found; falling back to chunked load to prevent stalls.")
            df = load_full_chunked(data_path, max_rows=400_000)
        else:
            df = load_small_dataframe(data_path)

    elif load_mode == "Chunked full file (safe subset)":
        df = load_full_chunked(data_path, max_rows=400_000)

    else:  # Direct full file
        st.info("Attempting to read entire file—this may take a long time or fail on Streamlit Cloud.")
        df = load_small_dataframe(data_path)  # uses same reader but won’t limit rows

except Exception as e:
    st.error("Failed to load data:")
    st.exception(e)
    st.stop()

st.success(f"Data loaded. Rows: {len(df):,} | Stores: {df['air_store_id'].nunique()}")
st.write(df.head())

# -------------------- Model --------------------
try:
    model = load_model()
    st.success("Model loaded.")
except Exception as e:
    st.error("Failed to load model:")
    st.exception(e)
    st.stop()

# -------------------- Prediction UI --------------------
stores = df["air_store_id"].unique()
if len(stores) == 0:
    st.warning("No stores found in data.")
    st.stop()

store_id = st.selectbox("Select restaurant (air_store_id):", options=stores, index=0)
store_df = df[df["air_store_id"] == store_id].sort_values("datetime_hour")
st.caption(f"Selected store rows: {len(store_df):,}")

if store_df.empty:
    st.warning("No rows for selected store.")
    st.stop()

candidate = build_next_row(store_df)
missing = candidate[[c for c in FEATURE_COLS if c in candidate]].isna()

if missing.any():
    st.warning(
        "Insufficient history to compute all features for the next hour "
        f"(missing: {', '.join(missing[missing].index.tolist())})."
    )
else:
    X_next = candidate[FEATURE_COLS].to_frame().T
    try:
        yhat = float(model.predict(X_next)[0])
        st.subheader("Next-hour forecast")
        st.write(f"Timestamp: **{candidate['datetime_hour']}**")
        st.write(f"Predicted orders: **{yhat:.2f}**")
    except Exception as e:
        st.error("Prediction failed:")
        st.exception(e)

# Recent context
st.subheader("Recent actuals (last 48 hours)")
if "hourly_orders" in store_df.columns:
    last_48 = store_df.tail(48).set_index("datetime_hour")[["hourly_orders"]]
    last_48.columns = ["actual_orders"]
    st.line_chart(last_48)
else:
    st.info("Column 'hourly_orders' not found; skipping chart.")
