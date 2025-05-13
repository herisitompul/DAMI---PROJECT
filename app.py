import streamlit as st
import numpy as np
import pickle
import folium
from keras.models import model_from_json
from streamlit_folium import st_folium

# =======================
# Load model dari .pkl
# =======================
with open("species_prediction_model1.pkl", "rb") as f:
    data = pickle.load(f)
    scaler = data["scaler"]
    model = model_from_json(data["model_architecture"])
    model.set_weights(data["model_weights"])
    features = data["features"]

# =======================
# UI Form
# =======================
st.set_page_config(page_title="Prediksi Spesies", layout="centered")
st.title("Prediksi Probabilitas Kehadiran Spesies")

with st.form("prediction_form"):
    lon = st.number_input("Longitude", format="%.6f")
    lat = st.number_input("Latitude", format="%.6f")
    year = st.number_input("Tahun", step=1, format="%d")
    geo = st.number_input("Nilai Geo Feature")
    speciesId = st.number_input("Species ID", step=1, format="%d")
    submitted = st.form_submit_button("Prediksi")

# =======================
# Hasil Prediksi
# =======================
if submitted:
    try:
        # Preprocessing
        input_data = np.array([[lon, lat, year, geo, speciesId]])
        input_scaled = scaler.transform(input_data)

        # Prediksi
        prob = model.predict(input_scaled)[0][0]
        st.success(f"✅ Prediksi Probabilitas Kehadiran Spesies: **{prob:.2f}**")

        # Peta Interaktif
        m = folium.Map(location=[lat, lon], zoom_start=6)
        folium.Marker(
            location=[lat, lon],
            popup=f"Lat: {lat}<br>Lon: {lon}<br>Probabilitas: {prob:.2f}",
            icon=folium.Icon(color='green')
        ).add_to(m)

        st.subheader("Peta Lokasi")
        st_folium(m, width=700, height=500)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")
