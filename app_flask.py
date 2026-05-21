# ============================================================
# SERVICIO WEB CON FLASK - Consumo del modelo Iris
# ============================================================
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Cargar modelo y etiquetas
MODEL_PATH = "modelo/iris_model.pkl"
NAMES_PATH = "modelo/target_names.pkl"

model        = joblib.load(MODEL_PATH)
target_names = joblib.load(NAMES_PATH)

# ------------------------------------------------------------
# RUTA RAÍZ - Info del servicio
# ------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "servicio": "Clasificador Iris",
        "version": "1.0",
        "endpoints": {
            "GET  /":         "Información del servicio",
            "GET  /health":   "Estado del servicio",
            "POST /predict":  "Realizar predicción"
        }
    })

# ------------------------------------------------------------
# RUTA /health - Estado del servicio
# ------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "modelo_cargado": model is not None})

# ------------------------------------------------------------
# RUTA /predict - Predicción
# Body esperado (JSON):
# {
#   "features": [5.1, 3.5, 1.4, 0.2]
# }
# ------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        if "features" not in data:
            return jsonify({"error": "Falta el campo 'features'"}), 400

        features = np.array(data["features"]).reshape(1, -1)

        if features.shape[1] != 4:
            return jsonify({"error": "Se requieren exactamente 4 features"}), 400

        prediction    = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        clase_nombre  = target_names[prediction]

        return jsonify({
            "prediccion_id":    int(prediction),
            "clase":            clase_nombre,
            "probabilidades": {
                name: round(float(prob), 4)
                for name, prob in zip(target_names, probabilities)
            },
            "features_recibidas": data["features"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------
# RUTA /predict/batch - Predicción múltiple
# ------------------------------------------------------------
@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    try:
        data = request.get_json(force=True)
        samples = np.array(data["samples"])
        predictions = model.predict(samples)

        return jsonify({
            "predicciones": [
                {"id": int(p), "clase": target_names[p]}
                for p in predictions
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Iniciando servicio Flask en http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
