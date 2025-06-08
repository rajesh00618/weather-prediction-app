import streamlit as st
import pandas as pd
import pickle
import datetime
import os
import gdown

# ------------------- Download from Google Drive if not present -------------------
file_ids = {
    "weather_model_desc.pkl": "1-uFeHnB0ZWMJi67izsSKARfO_7mVQzYu",
    "weather_label_encoder.pkl": "1nAkehTsaQbH10UCZqJiutoP7D0mi2qYK",
    "scaler.pkl": "1M2qIc31NOgOfljPvziABYOeH-aKSvU7E",
    "weather_model_dew.pkl": "1Rmwo0ixP3RGlExVRf-TcnAieJeoBGBGR",
    "weather_model_hum.pkl": "1ksYKUxWEAUFgAZViBWQtQeCgVI_yKG7I",
    "weather_model_max.pkl": "1zwhJfuk7nk4h2iTWRqdto1GnEHvaGOl-",
    "weather_model_min.pkl": "1jVxcwJv90cyiPq9-z4Uul5s6qfhWGaEp"
}

for filename, file_id in file_ids.items():
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)

# ------------------- Load models and tools -------------------
model_min = pickle.load(open("weather_model_min.pkl", "rb"))
model_max = pickle.load(open("weather_model_max.pkl", "rb"))
model_hum = pickle.load(open("weather_model_hum.pkl", "rb"))
model_dew = pickle.load(open("weather_model_dew.pkl", "rb"))
model_desc = pickle.load(open("weather_model_desc.pkl", "rb"))
le = pickle.load(open("weather_label_encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Weather Forecast", page_icon="⛅", layout="centered")
st.title("🌤️ Weather Forecast - 5 Day Prediction")
st.markdown("Enter today's weather to get the next 5 days forecast using ML models.")

# Input fields
date_input = st.date_input("Today's Date", datetime.date.today())
tempmin = st.number_input("Minimum Temperature (°C)", value=25.0)
tempmax = st.number_input("Maximum Temperature (°C)", value=35.0)
humidity = st.number_input("Humidity (%)", value=70.0)
dew = st.number_input("Dew Point (°C)", value=24.0)
desc = st.text_input("Current Description (optional)", value="Clear sky")

# ------------------- Forecast logic -------------------
if st.button("🔮 Predict 5-Day Forecast"):
    forecast = []
    current_date = date_input
    current_tempmin = tempmin
    current_tempmax = tempmax
    current_humidity = humidity
    current_dew = dew

    for i in range(5):
        current_date += datetime.timedelta(days=1)

        input_df = pd.DataFrame([{
            'tempmin': current_tempmin,
            'tempmax': current_tempmax,
            'humidity': current_humidity,
            'dew': current_dew
        }])

        input_scaled = scaler.transform(input_df)

        pred_min = model_min.predict(input_scaled)[0]
        pred_max = model_max.predict(input_scaled)[0]
        pred_hum = model_hum.predict(input_scaled)[0]
        pred_dew = model_dew.predict(input_scaled)[0]
        pred_desc_idx = model_desc.predict(input_scaled)[0]
        pred_desc = le.inverse_transform([int(pred_desc_idx)])[0]

        forecast.append({
            "date": current_date.strftime('%Y-%m-%d'),
            "min_temp": round(pred_min, 2),
            "max_temp": round(pred_max, 2),
            "humidity": round(pred_hum, 2),
            "dew": round(pred_dew, 2),
            "description": pred_desc
        })

        # Autoregressive update for next day
        current_tempmin = pred_min
        current_tempmax = pred_max
        current_humidity = pred_hum
        current_dew = pred_dew

    # ------------------- Display forecast -------------------
    st.markdown("### 📅 5-Day Weather Forecast")
    for day in forecast:
        st.write(f"**{day['date']}**")
        st.write(f"🌡️ Min: {day['min_temp']}°C | Max: {day['max_temp']}°C")
        st.write(f"💧 Humidity: {day['humidity']}% | Dew: {day['dew']}°C")
        st.write(f"🌈 Description: {day['description']}")
        st.markdown("---")
