import streamlit as st
import numpy as np
import joblib
import os

# ---------------------------
# Set page config FIRST
st.set_page_config(page_title="Amazon Delivery Time Predictor", layout="centered")

# ---------------------------
# Cache model & scaler for fast loading
@st.cache_resource
def load_model_and_scaler(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# ---------------------------
# Paths (local folder)
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "trained_models", "best_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "trained_models", "scaler.joblib")

# Load once
model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)

# ---------------------------
# Mapping dictionaries for categorical features
weather_map = {'Sunny':'Weather_Sunny','Cloudy':'Weather_Cloudy','Windy':'Weather_Windy','Stormy':'Weather_Stormy','Fog':'Weather_Fog','Sandstorms':'Weather_Sandstorms'}
traffic_map = {'Low':'Traffic_Low','Medium':'Traffic_Medium','High':'Traffic_High','Jam':'Traffic_Jam','Unknown':'Traffic_NaN'}
vehicle_map = {'motorcycle':'Vehicle_motorcycle','bicycle':'Vehicle_bicycle','scooter':'Vehicle_scooter','van':'Vehicle_van'}
area_map = {'Metropolitian':'Area_Metropolitian','Urban':'Area_Urban','Semi-Urban':'Area_Semi-Urban','Other':'Area_Other'}

# ---------------------------
# Prediction function
def predict_delivery(agent_age, agent_rating, distance, order_hour, processing_time,
                     weather, traffic, vehicle, area):
    feature_order = list(scaler.feature_names_in_)
    x = np.zeros(len(feature_order))

    # Fill numeric features
    num_features = ['Agent_Age','Agent_Rating','Distance','Order_Hour','Processing_Time']
    values = [agent_age, agent_rating, distance, order_hour, processing_time]
    for f, v in zip(num_features, values):
        matches = np.where(feature_order == f)[0]
        if len(matches) > 0:
            idx = matches[0]
            x[idx] = v
        else:
            print(f"Warning: numeric feature '{f}' not found in scaler.feature_names_in_")

    # Fill categorical features
    def set_feature(mapping, value):
        name = mapping.get(value)
        if name:
            matches = np.where(feature_order == name)[0]
            if len(matches) > 0:
                idx = matches[0]
                x[idx] = 1
            else:
                print(f"Warning: categorical feature '{name}' not found in scaler.feature_names_in_")

    set_feature(weather_map, weather)
    set_feature(traffic_map, traffic)
    set_feature(vehicle_map, vehicle)
    set_feature(area_map, area)

    # Scale and predict
    x_scaled = scaler.transform(x.reshape(1, -1))
    pred = model.predict(x_scaled)[0]
    return int(round(pred))

# ---------------------------
# Streamlit UI
st.title("🚚 Amazon Delivery Time Prediction (Fast Local)")

# Numeric inputs
Agent_Age = st.number_input("Agent Age", 18, 80, 30)
Agent_Rating = st.number_input("Agent Rating (0-5)", 0.0, 5.0, 4.5, 0.1)
Distance = st.number_input("Distance (km)", 0.0, 1000.0, 5.0, 0.1)
Order_Hour = st.slider("Order Hour (0-23)", 0, 23, 10)
Processing_Time = st.number_input("Processing Time (minutes)", 0.0, 500.0, 10.0, 0.5)

# Categorical inputs
weather = st.selectbox("Weather", list(weather_map.keys()))
traffic = st.selectbox("Traffic", list(traffic_map.keys()))
vehicle = st.selectbox("Vehicle Type", list(vehicle_map.keys()))
area = st.selectbox("Area Type", list(area_map.keys()))

# Predict button
if st.button("Predict"):
    try:
        pred_time = predict_delivery(Agent_Age, Agent_Rating, Distance, Order_Hour,
                                     Processing_Time, weather, traffic, vehicle, area)
        st.success(f"Estimated Delivery Time: {pred_time} minutes")
    except Exception as e:
        st.error(f"Prediction failed: {e}")