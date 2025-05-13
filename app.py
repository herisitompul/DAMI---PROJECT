from flask import Flask, render_template, request
import numpy as np
import pickle
import os
import folium
from keras.models import model_from_json, load_model

model = load_model("species_prediction_mlp.keras")

# Load model dari .pkl
with open("species_prediction_model1.pkl", "rb") as f:
    data = pickle.load(f)
    scaler = data["scaler"]
    model = model_from_json(data["model_architecture"])
    model.set_weights(data["model_weights"])
    features = data["features"]

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    map_file = None

    if request.method == "POST":
        try:
            # Ambil input dari form
            lon = float(request.form["lon"])
            lat = float(request.form["lat"])
            year = int(request.form["year"])
            geo = float(request.form["geo"])
            speciesId = int(request.form["speciesId"])

            # Preprocessing input
            input_data = np.array([[lon, lat, year, geo, speciesId]])
            input_scaled = scaler.transform(input_data)

            # Prediksi
            prob = model.predict(input_scaled)[0][0]
            prediction = f"Prediksi Probabilitas Kehadiran Spesies: {prob:.2f}"

            # Buat peta interaktif
            m = folium.Map(location=[lat, lon], zoom_start=6)
            folium.Marker(
                location=[lat, lon],
                popup=f"Lokasi Input<br>Lat: {lat}<br>Lon: {lon}<br>Probabilitas: {prob:.2f}",
                icon=folium.Icon(color='red')
            ).add_to(m)

            # Simpan sebagai HTML di folder static
            map_path = os.path.join("static", "map.html")
            m.save(map_path)
            map_file = "map.html"

        except Exception as e:
            prediction = f"Terjadi kesalahan: {e}"

    return render_template("index.html", prediction=prediction, map_file=map_file)

if __name__ == "__main__":
    app.run(debug=True)
