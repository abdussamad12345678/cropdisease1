import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from auth import register, login
from model import load_model
from utils import get_weather

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="PragyanAI", layout="wide")

# -------------------------------
# SESSION STATE
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -------------------------------
# LOGIN / REGISTER UI
# -------------------------------
menu = ["Login", "Register"]
choice = st.sidebar.selectbox("🔐 Account", menu)

if not st.session_state.logged_in:

    if choice == "Login":
        st.title("🔐 Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            success, msg = login(username, password)

            if success:
                st.session_state.logged_in = True
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    elif choice == "Register":
        st.title("📝 Register")

        new_user = st.text_input("Username")
        new_pass = st.text_input("Password", type="password")

        if st.button("Register"):
            success, msg = register(new_user, new_pass)

            if success:
                st.success(msg)
            else:
                st.error(msg)

    st.stop()

# -------------------------------
# LOGOUT
# -------------------------------
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

# -------------------------------
# HEADER
# -------------------------------
st.markdown("## 🌾 PragyanAI Crop Intelligence Dashboard")

# -------------------------------
# LOAD MODEL
# -------------------------------
model = load_model()

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("📊 Control Panel")
city = st.sidebar.text_input("📍 Location", "Delhi")
crop = st.sidebar.selectbox("🌾 Crop", ["Rice", "Wheat", "Corn"])
stage = st.sidebar.selectbox("🌱 Growth Stage", ["Seedling", "Vegetative", "Flowering", "Harvest"])

col1, col2, col3 = st.columns(3)

# -------------------------------
# ANALYZE BUTTON
# -------------------------------
if st.sidebar.button("🚀 Analyze Risk"):

    try:
        temp, humidity, rainfall = get_weather(city)
    except Exception as e:
        st.error(str(e))
        st.stop()

    col1.metric("🌡 Temperature", f"{temp} °C")
    col2.metric("💧 Humidity", f"{humidity}%")
    col3.metric("🌧 Rainfall", f"{rainfall} mm")

    dfi = (humidity * 0.5) + (rainfall * 0.3) + (temp * 0.2)

    st.markdown("### 🧠 Disease Favorability Index")
    st.progress(min(int(dfi), 100))

    prob = model.predict_proba([[temp, humidity, rainfall]])[0][1]

    st.markdown("### ⚠ Disease Risk Score")
    st.progress(int(prob * 100))

    if prob < 0.3:
        st.success("🟢 Low Risk")
    elif prob < 0.7:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")
        st.info("💊 Apply preventive spray")

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

st.markdown("---")
st.markdown("🚀 AI predicts crop disease before it happens")
