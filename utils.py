import requests

# 🔑 Apna API key yahan paste karo
API_KEY = "9f244592efe26bbd55cf0f9ddaeb63d6"

st.title("🌦️ Live City Weather App")

# Input
city = st.text_input("Enter City Name")

# Function to get weather
def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    
    response = requests.get(url)
    data = response.json()

    # Error Handling
    if data["cod"] != 200:
        return None

    weather = {
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "condition": data["weather"][0]["description"],
        "wind": data["wind"]["speed"]
    }

    return weather

# Button
if st.button("Get Weather"):
    if city:
        result = get_weather(city)

        if result:
            st.success(f"Weather in {city.title()}")
            st.write(f"🌡️ Temperature: {result['temp']} °C")
            st.write(f"💧 Humidity: {result['humidity']}%")
            st.write(f"☁️ Condition: {result['condition']}")
            st.write(f"🌬️ Wind Speed: {result['wind']} m/s")
        else:
            st.error("City not found or API issue")
    else:
        st.warning("Please enter a city name")
