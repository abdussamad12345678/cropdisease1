import requests

# 🔑 REPLACE THIS with your real API key
API_KEY = "9f244592efe26bbd55cf0f9ddaeb63d6"

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    try:
        response = requests.get(url)
        data = response.json()

        # ❌ Handle API errors properly
        if response.status_code != 200:
            raise Exception(data.get("message", "API error"))

        # ✅ Extract values safely
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]

        rainfall = 0
        if "rain" in data:
            rainfall = data["rain"].get("1h", 0)

        return temp, humidity, rainfall

    except Exception as e:
        raise Exception(f"Weather fetch failed: {str(e)}")
