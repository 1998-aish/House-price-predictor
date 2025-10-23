# app.py (fixed)
import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# --- Page config: MUST come before any st.* calls ---
st.set_page_config(page_title="House Price Predictor", layout="centered")

MODEL_PATH = "house_price_model.joblib"
CSV_PATH = "california_housing.csv"

def lightweight_train(csv_path=CSV_PATH, model_path=MODEL_PATH):
    """Train a small model quickly if no pre-trained model is found."""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    import joblib, time

    if not os.path.exists(csv_path):
        st.error(f"Dataset {csv_path} not found. Please upload it or commit it to your repo.")
        st.stop()

    st.info("Training a lightweight model (safe for Streamlit Cloud)...")
    t0 = time.time()
    df = pd.read_csv(csv_path)
    target = "MedHouseVal"
    X = df.drop(columns=[target])
    y = df[target]

    # smaller subset for faster training
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.9, random_state=42)
    model = RandomForestRegressor(n_estimators=20, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    st.success(f"Model trained and saved in {time.time()-t0:.1f}s!")
    return model

def load_or_train_model(path=MODEL_PATH):
    """Try to load existing model, otherwise train a lightweight one."""
    if os.path.exists(path):
        st.info("✅ Found trained model file.")
        return joblib.load(path)
    else:
        st.warning("⚠️ Model file not found.")
        return lightweight_train()

# Load or train model
model = load_or_train_model()

# --- Now safe to use st.* calls for UI and evaluation ---

# --- Evaluation: Predicted vs Actual (loads eval.csv saved by training) ---
eval_path = "eval.csv"
if os.path.exists(eval_path):
    eval_df = pd.read_csv(eval_path)
    # compute metrics (compatibly across sklearn versions)
    y_true = eval_df["y_true"].values
    y_pred = eval_df["y_pred"].values
    eval_rmse = np.sqrt(mean_squared_error(y_true, y_pred))   # <- compute RMSE this way
    eval_r2 = r2_score(y_true, y_pred)

    st.subheader("Model evaluation on test set")
    st.write(f"RMSE: {eval_rmse:.4f}  —  R²: {eval_r2:.4f}")

    # scatter plot predicted vs actual
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.4)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", linewidth=1)
    ax.set_xlabel("Actual median house value (100k USD units)")
    ax.set_ylabel("Predicted median house value (100k USD units)")
    ax.set_title("Predicted vs Actual (test set)")
    st.pyplot(fig)
else:
    st.info("Evaluation file eval.csv not found — run train_model.py to generate test-set predictions.")

# --- Page header and rest of UI ---
st.title("House Price Predictor (California Housing)")
st.markdown(
    """
This simple demo predicts median house value (in 100k USD units) using features from the California Housing dataset.
Move the sliders in the sidebar and click **Predict price**.
"""
)

# Feature importance block (kept as before)...
feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                 "Population", "AveOccup", "Latitude", "Longitude"]

def get_feature_importances(model, feature_names):
    try:
        importances = model.feature_importances_
        s = pd.Series(importances, index=feature_names)
        return s.sort_values(ascending=False)
    except Exception:
        try:
            coefs = model.coef_
            if hasattr(coefs, "ndim") and coefs.ndim > 1:
                coefs = coefs[0]
            s = pd.Series(abs(coefs), index=feature_names)
            return s.sort_values(ascending=False)
        except Exception:
            return None

fi = get_feature_importances(model, feature_names)
if fi is not None:
    st.subheader("Feature importance")
    st.write("Higher value → that feature influences the prediction more.")
    st.bar_chart(fi)
    st.table(fi.to_frame(name="importance"))
else:
    st.info("Feature importances not available for the loaded model.")

# Sidebar inputs
st.sidebar.header("Input features")
MedInc = st.sidebar.slider("Median income (tens of thousands)", 0.5, 15.0, 3.0, step=0.1)
HouseAge = st.sidebar.slider("House age (years)", 1, 100, 20)
AveRooms = st.sidebar.slider("Average rooms", 1.0, 20.0, 5.0, step=0.1)
AveBedrms = st.sidebar.slider("Average bedrooms", 0.5, 5.0, 1.0, step=0.1)
Population = st.sidebar.slider("Block population", 1, 5000, 1000)
AveOccup = st.sidebar.slider("Average occupants", 0.5, 20.0, 3.0, step=0.1)
Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 34.0, step=0.01)
Longitude = st.sidebar.slider("Longitude", -124.0, -114.0, -118.0, step=0.01)

input_dict = {
    "MedInc": MedInc,
    "HouseAge": HouseAge,
    "AveRooms": AveRooms,
    "AveBedrms": AveBedrms,
    "Population": Population,
    "AveOccup": AveOccup,
    "Latitude": Latitude,
    "Longitude": Longitude,
}
input_df = pd.DataFrame([input_dict])

st.subheader("Input features")
st.write(input_df)

# Prediction
if st.button("Predict price"):
    try:
        pred = model.predict(input_df)[0]
        st.subheader("Predicted median house value")
        st.write(f"{pred:.3f} (units: 100k USD)")
        st.info(f"Estimated price = ${pred * 100000:,.2f} USD")
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.write("Possible cause: model expects different columns/order. Re-run training if needed.")

st.write("---")
st.write("Tip: try different values and see how the prediction changes.")
