import requests

API_KEY = "6426773cbc70f50ad37ef85fb05d69d3"

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    
    res = requests.get(url)
    data = res.json()

    # ERROR HANDLING
    if "main" not in data:
        raise Exception(data.get("message", "Weather API Error"))

    temp = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    rainfall = data.get("rain", {}).get("1h", 0)

    return temp, humidity, rainfall
