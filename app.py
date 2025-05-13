import streamlit as st
import numpy as np
import pickle
import folium
from keras.models import model_from_json
from streamlit.components.v1 import html

# Load model dan scaler
with open("species_prediction_model1.pkl", "rb") as f:
    data = pickle.load(f)
    scaler = data["scaler"]
    model = model_from_json(data["model_architecture"])
    model.set_weights(data["model_weights"])
    features = data["features"]

# Konfigurasi Streamlit
st.set_page_config(page_title="Prediksi Spesies", layout="wide")
st.title("📍 Location-based Species Presence Prediction using CNN")

# Create two columns for input and output
col1, col2 = st.columns([1, 2])  # Adjust the ratio as needed

# Input Form in the left column
with col1:
    #st.header("Input Form")
    with st.form("species_form"):
        lon = st.text_input("Longitude:")
        lat = st.text_input("Latitude:")
        year = st.number_input("Year", min_value=0, step=1)
        geo = st.text_input("Geo Uncertainty (m):")
        speciesId = st.number_input("Species ID", min_value=0, step=1)
        submitted = st.form_submit_button("Predict")

# Output Map in the right column
with col2:
    st.header("🌍 Map of Location")
    if submitted:
        try:
            lon_float = float(lon)
            lat_float = float(lat)
            geo_float = float(geo)

            input_data = np.array([[lon_float, lat_float, year, geo_float, speciesId]])
            input_scaled = scaler.transform(input_data)
            prob = model.predict(input_scaled)[0][0]

            # Display the map
            m = folium.Map(location=[lat_float, lon_float], zoom_start=6)
            folium.Marker(
                location=[lat_float, lon_float],
                popup=f"Lat: {lat_float}, Lon: {lon_float}, Prob: {prob:.2f}",
                icon=folium.Icon(color='red')
            ).add_to(m)

            # Render map as HTML and display
            map_html = m._repr_html_()
            html(map_html, height=500, width=700)

        except ValueError:
            st.error("⚠️ Pastikan semua input numerik diisi dengan benar.")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")
    else:
        # Show placeholder message when no prediction or map yet
        st.markdown(
            """
            <h2 style="color:#6b7280; text-align:center; margin-top: 60px;">
            The prediction results and maps will appear here.
            </h2>
            """, unsafe_allow_html=True)

