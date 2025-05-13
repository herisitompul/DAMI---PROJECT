import streamlit as st
import numpy as np
import pickle
import folium
from keras.models import model_from_json
from keras.models import load_model
from streamlit_folium import st_folium

# Load model dan scaler
with open("species_prediction_model1.pkl", "rb") as f:
    data = pickle.load(f)
    scaler = data["scaler"]
    model = model_from_json(data["model_architecture"])
    model.set_weights(data["model_weights"])
    features = data["features"]

st.set_page_config(page_title="Prediksi Spesies", layout="wide")
st.title("📍 Location-based Species Presence Prediction using CNN")

# Input Form
with st.form("species_form"):
    lon = st.text_input("Longitude:")
    lat = st.text_input("Latitude:")
    year = st.number_input("Year", min_value=0, step=1)
    geo = st.text_input("Geo Uncertainty (m):")
    speciesId = st.number_input("Species ID", min_value=0, step=1)
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        lon = float(lon)
        lat = float(lat)
        geo = float(geo)

        input_data = np.array([[lon, lat, year, geo, speciesId]])
        input_scaled = scaler.transform(input_data)
        prob = model.predict(input_scaled)[0][0]

        st.success(f"✅ Prediksi Probabilitas Kehadiran Spesies: {prob:.2f}")

        # Buat peta folium
        m = folium.Map(location=[lat, lon], zoom_start=6)
        folium.Marker(
            location=[lat, lon],
            popup=f"Lat: {lat}, Lon: {lon}, Prob: {prob:.2f}",
            icon=folium.Icon(color='red')
        ).add_to(m)

        st.markdown("### 🌍 Map of Location")
        st_folium(m, width=700, height=500)

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
