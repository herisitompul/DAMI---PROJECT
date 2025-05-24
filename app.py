import streamlit as st
import pandas as pd
import joblib
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import time

# Load model dan scaler
model = joblib.load('model/final_rf_model.pkl')
scaler = joblib.load('model/final_scaler.pkl')

# Load dataset
df = pd.read_csv('data/dataset_sample_100000.csv')

st.title("Prediction of Species Presence")

# Form input user
lat = st.number_input("Latitude", format="%.6f")
lon = st.number_input("Longitude", format="%.6f")
elevation = st.number_input("Elevation", format="%.2f")
year = st.number_input("Tahun", format="%d", step=1)

# Tombol prediksi
if st.button("Prediksi"):
    with st.spinner('Memproses prediksi...'):
        time.sleep(2)

        # Cek apakah inputan ada di dataset
        match = df[
            (df['lat'] == lat) &
            (df['lon'] == lon) &
            (df['Elevation'] == elevation) &
            (df['year'] == year)
        ]

        if not match.empty:
            input_features = [[lat, lon, elevation, year]]
            input_scaled = scaler.transform(input_features)
            pred = model.predict(input_scaled)[0]

            st.session_state['prediksi'] = pred
            st.session_state['show_map'] = True
            st.session_state['map_lat'] = lat
            st.session_state['map_lon'] = lon
            st.session_state['map_elev'] = elevation
        else:
            st.warning("⚠️ Spesies di lokasi tersebut tidak ditemukan dalam data.")
            st.session_state['show_map'] = False

# Tampilkan hasil prediksi dan peta jika tersedia
if 'show_map' in st.session_state and st.session_state['show_map']:
    pred = st.session_state['prediksi']
    lat = st.session_state['map_lat']
    lon = st.session_state['map_lon']
    elev = st.session_state['map_elev']

    st.success(f"✅ Spesies {'Ditemukan (Label = 1)' if pred == 1 else 'Tidak Ditemukan (Label = 0)'} di lokasi ini.")

    # Buat teks popup yang menampilkan info lengkap
    popup_text = f"""
    <b>Latitude:</b> {lat:.5f}<br>
    <b>Longitude:</b> {lon:.5f}<br>
    <b>Elevation:</b> {elev:.2f} m<br>
    <b>Prediksi (Label):</b> {pred}
    """

    # Buat peta
    m = folium.Map(location=[lat, lon], zoom_start=6)
    color = 'green' if pred == 1 else 'red'
    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(popup_text, max_width=300)
    ).add_to(m)

    st_folium(m, width=700, height=500)
