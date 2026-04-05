import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import requests
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="PragyanAI", layout="wide")

DATA_FILE = "data.csv"
MODEL_FILE = "model.pkl"

# -------------------------------
# STYLING
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
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.title("🌾 PragyanAI Crop Intelligence Dashboard")

# -------------------------------
# MODEL FUNCTIONS
# -------------------------------
@st.cache_resource
def train_model():
    data = pd.read_csv(DATA_FILE)

    X = data[["temperature", "humidity", "rainfall"]]
    y = data["disease"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    return model


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return train_model()

    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


model = load_model()

# -------------------------------
# WEATHER API
# -------------------------------
API_KEY = st.secrets.get("API_KEY", "YOUR_API_KEY")

def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        res = requests.get(url, timeout=5)
        data = res.json()

        if data.get("cod") != 200:
            return None

        return {
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "rainfall": data.get("rain", {}).get("1h", 0)
        }

    except Exception:
        return None

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("📊 Control Panel")

city = st.sidebar.text_input("📍 Location", "Delhi")
crop = st.sidebar.selectbox("🌾 Crop", ["Rice", "Wheat", "Corn"])
stage = st.sidebar.selectbox("🌱 Growth Stage", ["Seedling", "Vegetative", "Flowering", "Harvest"])

analyze = st.sidebar.button("🚀 Analyze Risk")

# -------------------------------
# MAIN
# -------------------------------
if analyze:
    weather = get_weather(city)

    if not weather:
        st.error("❌ Unable to fetch weather data")
        st.stop()

    temp = weather["temp"]
    humidity = weather["humidity"]
    rainfall = weather["rainfall"]

    col1, col2, col3 = st.columns(3)

    col1.metric("🌡 Temperature", f"{temp} °C")
    col2.metric("💧 Humidity", f"{humidity}%")
    col3.metric("🌧 Rainfall", f"{rainfall} mm")

    # -------------------------------
    # DFI
    # -------------------------------
    dfi = (humidity * 0.5) + (rainfall * 0.3) + (temp * 0.2)

    st.subheader("🧠 Disease Favorability Index")
    st.progress(min(int(dfi), 100))

    # -------------------------------
    # PREDICTION
    # -------------------------------
    prob = model.predict_proba([[temp, humidity, rainfall]])[0][1]

    st.subheader("⚠ Disease Risk Score")
    st.progress(int(prob * 100))

    # -------------------------------
    # RISK LEVEL
    # -------------------------------
    if prob < 0.3:
        st.success("🟢 Low Risk")
    elif prob < 0.7:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")
        st.info("💊 Apply preventive measures within 2–3 days")

    # -------------------------------
    # WHAT-IF ANALYSIS
    # -------------------------------
    st.subheader("🔮 What-if Analysis")

    future_rain = rainfall + 10
    new_prob = model.predict_proba([[temp, humidity, future_rain]])[0][1]

    st.write(f"If rainfall increases by 10mm → Risk = {round(new_prob, 2)}")

# -------------------------------
# IMAGE ANALYSIS
# -------------------------------
st.subheader("📸 Leaf Disease Detection")

file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file)
    st.image(img, width=300)

    img_array = np.array(img)
    avg_pixel = img_array.mean()

    if avg_pixel < 100:
        st.error("⚠ Disease Detected")
    else:
        st.success("✅ Healthy Leaf")

# -------------------------------
# ANALYTICS
# -------------------------------
st.subheader("📊 Analytics Dashboard")

try:
    data = pd.read_csv(DATA_FILE)

    c1, c2 = st.columns(2)

    c1.line_chart(data[["temperature", "humidity", "rainfall"]])
    c2.bar_chart(data["disease"].value_counts())

except Exception:
    st.warning("⚠ Dataset not found or invalid")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("🚀 AI predicts crop disease before it happens")
