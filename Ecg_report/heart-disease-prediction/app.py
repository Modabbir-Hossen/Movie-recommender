"""
app.py
------
Flask web application for Heart Disease Prediction.

Routes:
    GET  /          → Dashboard with model metrics & plots
    POST /predict   → Accept form data, return prediction JSON
    GET  /api/metrics → Return model metrics as JSON
"""

import os
import sys
import json
import numpy as np
from flask import Flask, render_template, request, jsonify

# Ensure project root is on the path (works on Windows + Mac + Linux)
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.preprocess import FEATURE_COLS, FEATURE_INFO
from src.train       import load_artifacts

app = Flask(__name__)

# ─── Load model at startup ────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

try:
    model, scaler, metrics = load_artifacts(MODEL_DIR)
    MODEL_LOADED = True
    print(f"✅ Model loaded: {metrics['best_model']}")
except Exception as e:
    MODEL_LOADED = False
    model = scaler = metrics = None
    print(f"⚠️  Could not load model: {e}")
    print("   Run `python train_model.py` first.")


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    ctx = {
        "model_loaded": MODEL_LOADED,
        "metrics":      metrics,
        "feature_info": FEATURE_INFO,
        "plots": {
            "cv":      _plot_exists("cv_comparison"),
            "cm":      _plot_exists("confusion_matrix"),
            "roc":     _plot_exists("roc_curve"),
            "feat":    _plot_exists("feature_importance"),
        }
    }
    return render_template("index.html", **ctx)


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    try:
        data = request.get_json(force=True)

        # Extract & validate features
        features = []
        for col in FEATURE_COLS:
            val = data.get(col)
            if val is None:
                return jsonify({"error": f"Missing field: {col}"}), 400
            features.append(float(val))

        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)

        pred       = int(model.predict(X_scaled)[0])
        prob       = float(model.predict_proba(X_scaled)[0][pred])
        prob_pos   = float(model.predict_proba(X_scaled)[0][1])

        risk_level = (
            "Low"    if prob_pos < 0.35 else
            "Medium" if prob_pos < 0.65 else
            "High"
        )

        result = {
            "prediction":  pred,
            "label":       "Heart Disease Detected" if pred == 1 else "No Heart Disease",
            "probability": round(prob_pos * 100, 1),
            "risk_level":  risk_level,
            "model_used":  metrics["best_model"],
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/metrics")
def api_metrics():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify(metrics)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _plot_exists(name: str) -> bool:
    path = os.path.join(os.path.dirname(__file__), "static", "plots", f"{name}.png")
    return os.path.exists(path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
