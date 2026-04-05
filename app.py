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
# STYLE
# -------------------------------
st.markdown("""
<style>
.main { background-color: #f5f7fa; }
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.title("🌾 PragyanAI Crop Intelligence Dashboard")

# -------------------------------
# MODEL
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
    return pickle.load(open(MODEL_FILE, "rb"))

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
    except:
        return None

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("📊 Control Panel")

city = st.sidebar.text_input("📍 Location", "Delhi")
crop = st.sidebar.selectbox("🌾 Crop", ["Rice", "Wheat", "Corn"])
stage = st.sidebar.selectbox("🌱 Growth Stage", ["Seedling", "Vegetative", "Flowering", "Harvest"])

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["🌤 Prediction", "📊 Analytics", "📸 Leaf AI"])

# ===============================
# 🌤 PREDICTION TAB
# ===============================
with tab1:

    st.subheader("🌤 Weather & Risk Prediction")

    mode = st.radio("Select Mode", ["Auto (Live Weather)", "Manual Simulation"])

    if mode == "Auto (Live Weather)":
        weather = get_weather(city)

        if not weather:
            st.error("❌ Unable to fetch weather data")
            st.stop()

        temp = weather["temp"]
        humidity = weather["humidity"]
        rainfall = weather["rainfall"]

    else:
        st.info("🎛 Adjust values manually")

        temp = st.slider("🌡 Temperature (°C)", 0, 50, 25)
        humidity = st.slider("💧 Humidity (%)", 0, 100, 60)
        rainfall = st.slider("🌧 Rainfall (mm)", 0, 200, 20)

    col1, col2, col3 = st.columns(3)

    col1.metric("🌡 Temperature", f"{temp} °C")
    col2.metric("💧 Humidity", f"{humidity}%")
    col3.metric("🌧 Rainfall", f"{rainfall} mm")

    # DFI
    dfi = (humidity * 0.5) + (rainfall * 0.3) + (temp * 0.2)

    st.subheader("🧠 Disease Favorability Index")
    st.progress(min(int(dfi), 100))

    # Prediction
    prob = model.predict_proba([[temp, humidity, rainfall]])[0][1]

    st.subheader("⚠ Disease Risk Score")
    st.progress(int(prob * 100))

    if prob < 0.3:
        st.success("🟢 Low Risk")
    elif prob < 0.7:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")
        st.info("💊 Apply preventive measures")

    # Simulation Graph
    st.subheader("📈 Risk Simulation")

    sim_rain = np.linspace(0, rainfall + 50, 20)
    sim_prob = [
        model.predict_proba([[temp, humidity, r]])[0][1]
        for r in sim_rain
    ]

    sim_df = pd.DataFrame({
        "Rainfall": sim_rain,
        "Risk": sim_prob
    })

    st.line_chart(sim_df.set_index("Rainfall"))

# ===============================
# 📊 ANALYTICS TAB
# ===============================
with tab2:

    st.subheader("📊 Dataset Insights")

    try:
        data = pd.read_csv(DATA_FILE)

        st.write("### Preview")
        st.dataframe(data.head())

        st.write("### Feature Trends")
        st.line_chart(data[["temperature", "humidity", "rainfall"]])

        st.write("### Disease Distribution")
        st.bar_chart(data["disease"].value_counts())

        st.write("### 🔍 Filter Data")

        temp_range = st.slider("Temperature Range", 0, 50, (10, 40))

        filtered = data[
            (data["temperature"] >= temp_range[0]) &
            (data["temperature"] <= temp_range[1])
        ]

        st.write(f"Filtered Rows: {len(filtered)}")
        st.dataframe(filtered)

    except:
        st.warning("⚠ Dataset not found")

# ===============================
# 📸 LEAF AI TAB
# ===============================
with tab3:

    st.subheader("📸 Leaf Disease Detection")

    file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

    if file:
        img = Image.open(file)

        col1, col2 = st.columns(2)

        col1.image(img, caption="Uploaded Image")

        img_array = np.array(img)
        avg_pixel = img_array.mean()

        with col2:
            st.write("### Analysis Result")

            if avg_pixel < 100:
                st.error("⚠ Disease Detected")
            else:
                st.success("✅ Healthy Leaf")

            st.write(f"Pixel Score: {round(avg_pixel,2)}")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("🚀 AI predicts crop disease before it happens")
