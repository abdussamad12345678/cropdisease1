import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from auth import register, login, get_security_question, reset_password
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
# LOGIN / REGISTER / FORGOT
# -------------------------------
if not st.session_state.logged_in:

    # 🎨 Yellow Theme
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #F4C542;
    }
    h1, h2, h3, label { color: #1C1C1C; }
    input { background-color: white !important; border-radius: 8px !important; }
    .stButton > button {
        background-color: #0E7C1F;
        color: white;
        border-radius: 10px;
        height: 45px;
        width: 100%;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🌾 PragyanAI Login Portal")

    tab1, tab2, tab3 = st.tabs(["🔐 Login", "📝 Register", "🔁 Forgot Password"])

    # -------- LOGIN --------
    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login", key="login_btn"):
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

        question = st.selectbox("Security Question", [
            "Your pet name?",
            "Your school name?",
            "Your favorite color?"
        ], key="reg_question")

        answer = st.text_input("Answer", key="reg_answer")

        if st.button("Register", key="register_btn"):
            success, msg = register(new_user, new_pass, question, answer)
            if success:
                st.success("✅ Registered! Now login.")
            else:
                st.error(msg)

    # -------- FORGOT PASSWORD --------
    with tab3:
        f_user = st.text_input("Enter Username", key="forgot_user")

        if st.button("Get Question", key="get_q_btn"):
            q = get_security_question(f_user)

            if q:
                st.session_state.reset_user = f_user
                st.session_state.question = q
            else:
                st.error("User not found")

        if "question" in st.session_state:
            st.info(st.session_state.question)

            ans = st.text_input("Answer", key="forgot_answer")
            new_pass = st.text_input("New Password", type="password", key="forgot_new_pass")

            if st.button("Reset Password", key="reset_btn"):
                success, msg = reset_password(
                    st.session_state.reset_user,
                    ans,
                    new_pass
                )

                if success:
                    st.success(msg)
                    del st.session_state.question
                else:
                    st.error(msg)

    st.stop()

# -------------------------------
# DASHBOARD (NORMAL)
# -------------------------------
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
if st.sidebar.button("🚪 Logout", key="logout_btn"):
    st.session_state.logged_in = False
    st.rerun()

# -------------------------------
# HEADER
# -------------------------------
st.markdown("## 🌾 PragyanAI Crop Intelligence Dashboard")

model = load_model()

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("📊 Control Panel")
city = st.sidebar.text_input("📍 Location", "Delhi", key="city")
crop = st.sidebar.selectbox("🌾 Crop", ["Rice", "Wheat", "Corn"], key="crop")
stage = st.sidebar.selectbox("🌱 Growth Stage", ["Seedling", "Vegetative", "Flowering", "Harvest"], key="stage")

col1, col2, col3 = st.columns(3)

# -------------------------------
# ANALYZE
# -------------------------------
if st.sidebar.button("🚀 Analyze Risk", key="analyze_btn"):

    try:
        temp, humidity, rainfall = get_weather(city)
    except Exception as e:
        st.error(str(e))
        st.stop()

    col1.metric("🌡 Temperature", f"{temp} °C")
    col2.metric("💧 Humidity", f"{humidity}%")
    col3.metric("🌧 Rainfall", f"{rainfall} mm")

    dfi = (humidity * 0.5) + (rainfall * 0.3) + (temp * 0.2)
    st.progress(min(int(dfi), 100))

    prob = model.predict_proba([[temp, humidity, rainfall]])[0][1]
    st.progress(int(prob * 100))

    if prob < 0.3:
        st.success("🟢 Low Risk")
    elif prob < 0.7:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

# -------------------------------
# IMAGE
# -------------------------------
st.markdown("### 📸 Leaf Disease Detection")

file = st.file_uploader("Upload Leaf Image", key="file_upload")

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

