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

        # Cek spesies berdasarkan lokasi (lat, lon) saja
        location_matches = df[
            (df['lat'] == lat) &
            (df['lon'] == lon)
        ]

        if not location_matches.empty:
            # Buat prediksi untuk input user
            input_features = [[lat, lon, elevation, year]]
            input_scaled = scaler.transform(input_features)
            user_pred = model.predict(input_scaled)[0]

            # Simpan hasil ke session state
            st.session_state['user_prediksi'] = user_pred
            st.session_state['location_data'] = location_matches
            st.session_state['show_map'] = True
            st.session_state['map_lat'] = lat
            st.session_state['map_lon'] = lon
            st.session_state['map_elev'] = elevation
            st.session_state['input_year'] = year
        else:
            st.warning("⚠️ Tidak ada data spesies di lokasi tersebut.")
            st.session_state['show_map'] = False

# Tampilkan hasil prediksi dan peta jika tersedia
if 'show_map' in st.session_state and st.session_state['show_map']:
    user_pred = st.session_state['user_prediksi']
    location_data = st.session_state['location_data']
    lat = st.session_state['map_lat']
    lon = st.session_state['map_lon']
    elev = st.session_state['map_elev']
    input_year = st.session_state['input_year']

    # Tampilkan prediksi untuk input user
    st.success(f"✅ Prediksi: Spesies {'Ditemukan (Label = 1)' if user_pred == 1 else 'Tidak Ditemukan (Label = 0)'}")

    # Tampilkan informasi semua spesies di lokasi yang sama
    st.subheader(f"📍 Semua Spesies di Lokasi ({lat:.5f}, {lon:.5f})")
    
    # Tampilkan tabel dengan kolom yang dipilih saja
    st.write(f"**Ada {len(location_data)} spesies ditemukan di lokasi ini:**")
    
    # Pilih kolom yang akan ditampilkan
    display_columns = []
    if 'speciesId' in location_data.columns:
        display_columns.append('speciesId')
    display_columns.extend(['Elevation', 'year'])
    if 'label' in location_data.columns:
        display_columns.append('label')
    
    # Tampilkan hanya kolom yang diperlukan dan reset index
    display_data = location_data[display_columns].copy().reset_index(drop=True)
    
    # Rename kolom agar lebih jelas
    column_rename = {
        'speciesId': 'Species ID',
        'Elevation': 'Elevation (m)',
        'year': 'Year',
        'label': 'Label'
    }
    display_data = display_data.rename(columns=column_rename)
    
    st.dataframe(display_data, use_container_width=True)

    # Buat peta dengan semua data di lokasi yang sama (satu titik)
    m = folium.Map(location=[lat, lon], zoom_start=12)
    
    # Buat satu popup yang berisi semua informasi
    all_species_info = []
    
    # Tambahkan info prediksi user
    user_info = f"""
    <div style='margin-bottom: 15px; padding: 10px; border: 2px solid blue; border-radius: 5px; background-color: #e6f3ff;'>
    <b>🔮 Hasil Prediksi</b><br>
    <b>Latitude:</b> {lat:.5f}<br>
    <b>Longitude:</b> {lon:.5f}<br>
    <b>Elevation:</b> {elev:.2f} m<br>
    <b>Tahun:</b> {input_year}<br>
    <b>Prediksi:</b> {'Ditemukan (1)' if user_pred == 1 else 'Tidak Ditemukan (0)'}
    </div>
    """
    
    # Tambahkan info setiap spesies existing
    existing_info = "<b>📊 DATA EXISTING DI LOKASI INI:</b><br><br>"
    
    for idx, row in location_data.iterrows():
        speciesId = row['speciesId'] if 'speciesId' in row else 'N/A'
        species_name = row['species'] if 'species' in row else 'N/A'
        label = int(row['label']) if 'label' in row else 'N/A'
        elevation = row['Elevation']
        year = int(row['year'])
        
        # Warna berdasarkan label
        if label == 1:
            color_style = "color: green; font-weight: bold;"
            status = "✅ Ditemukan"
        elif label == 0:
            color_style = "color: red; font-weight: bold;"
            status = "❌ Tidak Ditemukan"
        else:
            color_style = "color: gray;"
            status = "❓ Tidak Diketahui"
        
        existing_info += f"""
        <div style='margin: 5px 0; padding: 8px; border-left: 3px solid {"green" if label == 1 else "red" if label == 0 else "gray"}; background-color: #f9f9f9;'>
        <b>Species ID:</b> {speciesId} | <b>Elevation:</b> {elevation:.0f}m | <b>Tahun:</b> {year}<br>
        <span style='{color_style}'>Status: {status}</span>
        </div>
        """
    
    # Gabungkan semua info
    combined_popup = user_info + existing_info
    
    # Tambahkan marker tunggal dengan semua informasi
    folium.Marker(
        location=[lat, lon],
        icon=folium.Icon(color='blue', icon='info-sign', prefix='glyphicon'),
        popup=folium.Popup(combined_popup, max_width=400, max_height=400)
    ).add_to(m)
    
    # Tambahkan circle untuk menunjukkan area lokasi
    folium.Circle(
        location=[lat, lon],
        radius=100,  # 100 meter radius
        color='blue',
        fill=True,
        fillColor='blue',
        fillOpacity=0.1,
        popup=f"Area lokasi penelitian<br>Radius: 100m"
    ).add_to(m)

    # Tampilkan legend
    st.markdown("""    
    *Klik marker untuk melihat detail lengkap semua spesies di lokasi ini*
    """)

    st_folium(m, width=700, height=500)
