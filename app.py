import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import requests

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="PragyanAI", layout="wide")

# -------------------------------
# CUSTOM CSS (DASHING UI)
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
}
.metric-box {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("## 🌾 PragyanAI Crop Intelligence Dashboard")

# -------------------------------
# MODEL
# -------------------------------
MODEL_FILE = "model.pkl"

def train_model():
    data = pd.read_csv("data.csv")
    X = data[["temperature", "humidity", "rainfall"]]
    y = data["disease"]

    model = RandomForestClassifier()
    model.fit(X, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    return model

def load_model():
    if not os.path.exists(MODEL_FILE):
        return train_model()
    return pickle.load(open(MODEL_FILE, "rb"))

model = load_model()

# -------------------------------
# REAL WEATHER (API)
# -------------------------------
API_KEY = "9f244592efe26bbd55cf0f9ddaeb63d6"

def get_real_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    
    response = requests.get(url)
    data = response.json()

    if data["cod"] != 200:
        return None, None, None

    temp = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    rainfall = data.get("rain", {}).get("1h", 0)

    return temp, humidity, rainfall

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("📊 Control Panel")
city = st.sidebar.text_input("📍 Location", "Delhi")
crop = st.sidebar.selectbox("🌾 Crop", ["Rice", "Wheat", "Corn"])
stage = st.sidebar.selectbox("🌱 Growth Stage", ["Seedling", "Vegetative", "Flowering", "Harvest"])

# -------------------------------
# MAIN GRID
# -------------------------------
col1, col2, col3 = st.columns(3)

# -------------------------------
# BUTTON ACTION
# -------------------------------
if st.sidebar.button("🚀 Analyze Risk"):

    temp, humidity, rainfall = get_real_weather(city)

    # SAFETY CHECK
    if temp is None:
        st.error("City not found or API error")
        st.stop()

    # METRICS
    col1.metric("🌡 Temperature", f"{temp} °C")
    col2.metric("💧 Humidity", f"{humidity}%")
    col3.metric("🌧 Rainfall", f"{rainfall} mm")

    # DFI
    dfi = (humidity * 0.5) + (rainfall * 0.3) + (temp * 0.2)

    st.markdown("### 🧠 Disease Favorability Index")
    st.progress(min(int(dfi),100))

    # PREDICTION
    prob = model.predict_proba([[temp, humidity, rainfall]])[0][1]

    st.markdown("### ⚠ Disease Risk Score")
    st.progress(int(prob * 100))

    # RISK LEVEL
    if prob < 0.3:
        st.success("🟢 Low Risk")
    elif prob < 0.7:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")
        st.info("💊 Apply preventive spray in 2–3 days")

    # WHAT IF
    st.markdown("### 🔮 What-if Analysis")
    new_prob = model.predict_proba([[temp, humidity, rainfall + 10]])[0][1]
    st.write(f"If rainfall increases → Risk = {round(new_prob,2)}")

# -------------------------------
# IMAGE SECTION
# -------------------------------
st.markdown("### 📸 Leaf Disease Detection")

file = st.file_uploader("Upload Leaf Image")

if file:
    img = Image.open(file)
    st.image(img, width=300)

    avg = np.array(img).mean()
    if avg < 100:
        st.error("Disease Detected")
    else:
        st.success("Healthy Leaf")

# -------------------------------
# DASHBOARD
# -------------------------------
st.markdown("### 📊 Analytics Dashboard")

data = pd.read_csv("data.csv")

c1, c2 = st.columns(2)
c1.line_chart(data[["temperature", "humidity", "rainfall"]])
c2.bar_chart(data["disease"].value_counts())

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("🚀 AI predicts crop disease before it happens")
