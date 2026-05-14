"""
app.py — Flask REST API for Movie Recommender
Run: python backend/app.py
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__, static_folder="../static", template_folder="..")
CORS(app)

# ── LOAD MODEL ────────────────────────────────────────────────────────────────

PKL_PATH = "backend/recommender.pkl"

try:
    from backend.recommender import MovieRecommender
    if os.path.exists(PKL_PATH):
        recommender = MovieRecommender.load(PKL_PATH)
        print("✅ Model loaded from cache.")
    else:
        print("⚙️  Building model (first run — this takes ~30s)…")
        recommender = MovieRecommender.build_and_save()
except Exception as e:
    print(f"⚠️  Model unavailable: {e}")
    recommender = None


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("..", "index.html")


@app.route("/api/search")
def search():
    q = request.args.get("q", "").strip()
    if not recommender or len(q) < 2:
        return jsonify([])
    results = recommender.movies_df[
        recommender.movies_df["title"].str.lower().str.contains(q.lower(), na=False)
    ][["title", "year", "genres"]].head(8)
    return jsonify(results.to_dict(orient="records"))


@app.route("/api/recommend")
def recommend():
    title   = request.args.get("title", "").strip()
    mode    = request.args.get("mode", "content")   # content | collab | hybrid
    user_id = request.args.get("user_id", "1")
    n       = int(request.args.get("n", 10))

    if not recommender or not title:
        return jsonify({"error": "Model unavailable or no title provided"}), 400

    if mode == "content":
        recs = recommender.get_content_recommendations(title, n)
    elif mode == "collab":
        recs = recommender.get_collaborative_recommendations(user_id, n)
    elif mode == "hybrid":
        recs = recommender.get_hybrid_recommendations(title, user_id, n)
    else:
        return jsonify({"error": "mode must be content | collab | hybrid"}), 400

    return jsonify({"title": title, "mode": mode, "recommendations": recs})


@app.route("/api/movies")
def all_movies():
    if not recommender:
        return jsonify([])
    cols = ["title", "year", "genres", "vote_average"]
    return jsonify(recommender.movies_df[cols].head(500).to_dict(orient="records"))


@app.route("/api/status")
def status():
    return jsonify({
        "status": "ok",
        "model_loaded": recommender is not None,
        "movies": len(recommender.movies_df) if recommender else 0,
        "svd_available": recommender.svd_model is not None if recommender else False
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
