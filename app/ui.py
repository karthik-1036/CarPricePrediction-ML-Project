import streamlit as st
import pandas as pd
import yaml
import joblib

# Load trained model
model = joblib.load("saved_models/car_price_model_xgb.pkl")

# Load YAML schema
with open("app/input_schema.yaml", "r") as f:
    schema = yaml.safe_load(f)

st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Used Car Price Predictor (Full Version)")
st.write("Fill in all car details to get the most accurate resale price estimate.")

# Form for inputs
with st.form("predict_form"):
    user_input = {}
    for field, config in schema.items():
        if config["type"] == "categorical":
            user_input[field] = st.selectbox(field, config["options"])
        elif config["type"] == "numerical":
            r = config["range"]
            # int or float range
            if isinstance(r[0], int):
                user_input[field] = st.slider(field, r[0], r[1], value=int((r[0] + r[1]) // 2))
            else:
                user_input[field] = st.slider(field, float(r[0]), float(r[1]), value=round((r[0] + r[1]) / 2, 1))
    
    submit = st.form_submit_button("Predict Price 💰")

# Run prediction
if submit:
    df = pd.DataFrame([user_input])
    try:
        prediction = model.predict(df)[0]
        st.success(f"Estimated Resale Price: ₹{int(prediction):,}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
