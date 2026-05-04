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
# LOGIN / REGISTER SCREEN
# -------------------------------
if not st.session_state.logged_in:

    # 🎨 Blinkit Theme ONLY for login page
    st.markdown("""
    <style>

    [data-testid="stAppViewContainer"] {
        background-color: #F4C542;
    }

    h1, h2, h3, label {
        color: #1C1C1C;
    }

    input {
        background-color: white !important;
        border-radius: 8px !important;
    }

    .stButton > button {
        background-color: #0E7C1F;
        color: white;
        border-radius: 10px;
        height: 45px;
        width: 100%;
        font-weight: bold;
    }

    button[data-baseweb="tab"] {
        color: #1C1C1C;
        font-weight: bold;
    }

    </style>
    """, unsafe_allow_html=True)

    st.title("🌾 PragyanAI Login Portal")

    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

    # -------- LOGIN --------
    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            success, msg = login(username, password)

            if success:
                st.session_state.logged_in = True
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    # -------- REGISTER --------
    with tab2:
        new_user = st.text_input("Create Username", key="reg_user")
        new_pass = st.text_input("Create Password", type="password", key="reg_pass")

        if st.button("Register"):
            success, msg = register(new_user, new_pass)

            if success:
                st.success("✅ Registration successful! Go to Login tab.")
            else:
                st.error(msg)

    st.stop()

# -------------------------------
# NORMAL DASHBOARD (NO YELLOW THEME)
# -------------------------------

# Reset background to default after login
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f5f7fa;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOGOUT
# -------------------------------
st.sidebar.title("🔐 Account")
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
        st.info("💊 Apply preventive spray in 2–3 days")

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
