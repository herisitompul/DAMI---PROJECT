import streamlit as st
import pandas as pd
import joblib
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import numpy as np
import time

# Load model dan scaler
model = joblib.load('model/final_rf_model.pkl')
scaler = joblib.load('model/final_scaler.pkl')

# Load dataset
df = pd.read_csv('data/dataset_sample_100000.csv')

# Ambil semua label/spesies dari training data
label_columns = [col for col in df.columns if col.startswith('label_') or col == 'label']
multi_label_mode = len(label_columns) > 1

# Rentang tahun valid
min_year, max_year = df['year'].min(), df['year'].max()

st.title("Prediction of Species Presence")

# Form input user
lat = st.number_input("Latitude", format="%.6f")
lon = st.number_input("Longitude", format="%.6f")
elevation = st.number_input("Elevation", format="%.2f")
year = st.number_input("Tahun", format="%d", step=1)

# Cek validitas tahun
is_year_valid = min_year <= year <= max_year
if not is_year_valid:
    st.warning(f"⚠️ Tahun {year} berada di luar rentang data pelatihan ({min_year} - {max_year})")

# Tombol prediksi
if st.button("Prediksi"):
    if not is_year_valid:
        st.error("❌ Input tahun tidak valid. Silakan masukkan tahun antara 2017 dan 2021.")
        st.session_state['show_map'] = False
        st.stop()  # Hentikan eksekusi jika tahun tidak valid
    else:
        with st.spinner('Memproses prediksi...'):
            time.sleep(2)

            # Filter berdasarkan lat, lon, dan tahun
            location_matches = df[(df['lat'] == lat) & (df['lon'] == lon) & (df['year'] == year)]

            input_features = [[lat, lon, elevation, year]]
            input_scaled = scaler.transform(input_features)
            pred = model.predict(input_scaled)

            # Multi-label prediction
            if pred.ndim > 1 and pred.shape[1] > 1:
                pred = pred[0]
                predicted_labels = np.where(pred == 1)[0]
                st.session_state['user_prediction'] = pred.tolist()
                st.session_state['predicted_indices'] = predicted_labels.tolist()
            else:
                pred = int(pred[0])
                st.session_state['user_prediction'] = [pred]
                st.session_state['predicted_indices'] = [0] if pred == 1 else []

            st.session_state['location_data'] = location_matches
            st.session_state['show_map'] = True
            st.session_state['input'] = {
                'lat': lat, 'lon': lon, 'elevation': elevation, 'year': year
            }

# Tampilkan hasil prediksi dan visualisasi
if st.session_state.get('show_map', False):
    lat = st.session_state['input']['lat']
    lon = st.session_state['input']['lon']
    elev = st.session_state['input']['elevation']
    year = st.session_state['input']['year']
    pred = st.session_state['user_prediction']
    predicted_indices = st.session_state['predicted_indices']
    location_data = st.session_state['location_data']

    if predicted_indices:
        st.success(f"✅ Model memprediksi {len(predicted_indices)} spesies hadir di lokasi ini.")
    else:
        st.error("❌ Tidak ada spesies diprediksi hadir oleh model.")

    st.subheader(f"📍 Data Observasi Historis di Lokasi ({lat:.5f}, {lon:.5f})")
    if not location_data.empty:
        st.write(f"Ada {len(location_data)} baris observasi historis di lokasi ini:")
        display_columns = []
        if 'speciesId' in location_data.columns:
            display_columns.append('speciesId')
        display_columns.extend(['Elevation', 'year'])
        if 'label' in location_data.columns:
            display_columns.append('label')
        if 'species' in location_data.columns:
            display_columns.append('species')

        location_data_disp = location_data[display_columns].copy()
        location_data_disp = location_data_disp.rename(columns={
            'speciesId': 'Species ID',
            'Elevation': 'Elevation (m)',
            'year': 'Year',
            'label': 'Label',
            'species': 'Species Name'
        })
        st.dataframe(location_data_disp, use_container_width=True)
    else:
        st.info("🔍 Tidak ada data observasi di lokasi ini untuk tahun tersebut.")

    # Tampilkan spesies hasil prediksi
    st.subheader("🔮 Spesies yang Diprediksi Hadir oleh Model")
    if multi_label_mode:
        species_map = df[['speciesId', 'species']].drop_duplicates().reset_index(drop=True)
        species_map = species_map.reset_index().rename(columns={'index': 'model_index'})

        predicted_species = species_map.iloc[predicted_indices] if len(predicted_indices) > 0 else pd.DataFrame()
        if not predicted_species.empty:
            st.dataframe(predicted_species.rename(columns={
                'speciesId': 'Predicted Species ID',
                'species': 'Predicted Species Name'
            }), use_container_width=True)
        else:
            st.write("Model tidak memprediksi adanya spesies yang hadir.")
    else:
        st.write("Hasil prediksi: ", "✅ Spesies hadir" if pred[0] == 1 else "❌ Spesies tidak hadir")

    # Peta visualisasi
    m = folium.Map(location=[lat, lon], zoom_start=12)
    popup_info = f"""
    <b>Prediksi Model</b><br>
    Latitude: {lat:.5f}<br>
    Longitude: {lon:.5f}<br>
    Elevation: {elev:.2f} m<br>
    Tahun: {year}<br>
    Jumlah spesies diprediksi hadir: {len(predicted_indices)}
    """
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_info, max_width=300),
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

    folium.Circle(
        location=[lat, lon],
        radius=100,
        color='blue',
        fill=True,
        fillColor='blue',
        fillOpacity=0.1
    ).add_to(m)

    st_folium(m, width=700, height=500)
