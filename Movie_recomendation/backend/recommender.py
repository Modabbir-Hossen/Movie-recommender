"""
recommender.py — Core ML recommendation engine
TMDB 5000 Movies · Content-Based + SVD Collaborative
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
import os
import pickle


class MovieRecommender:
    def __init__(self, data_dir="data"):
        self.data_dir     = data_dir
        self.movies_df    = None
        self.sim_matrix   = None
        self.indices      = None
        self.svd_model    = None
        self._is_fitted   = False

    # ── DATA LOADING ────────────────────────────────────────────────────────

    def load_data(self):
        movies_path  = os.path.join(self.data_dir, "tmdb_5000_movies.csv")
        credits_path = os.path.join(self.data_dir, "tmdb_5000_credits.csv")

        movies  = pd.read_csv(movies_path)
        credits = pd.read_csv(credits_path)

        # Merge on title
        credits.rename(columns={"movie_id": "id"}, inplace=True)
        df = movies.merge(credits, on="id")

        # Keep relevant columns
        df = df[["id", "title", "overview", "genres", "keywords",
                 "cast", "crew", "vote_average", "vote_count",
                 "release_date", "popularity"]].copy()

        # Parse JSON columns
        for col in ["genres", "keywords"]:
            df[col] = df[col].apply(self._extract_names)
        df["cast"]  = df["cast"].apply(lambda x: self._extract_names(x, limit=5))
        df["crew"]  = df["crew"].apply(self._extract_director)

        # Fill nulls
        df["overview"].fillna("", inplace=True)

        # Extract year
        df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)

        self.movies_df = df.reset_index(drop=True)
        print(f"Loaded {len(self.movies_df)} movies.")
        return self

    # ── FEATURE ENGINEERING ─────────────────────────────────────────────────

    @staticmethod
    def _extract_names(obj_str, limit=None):
        try:
            objs = literal_eval(obj_str)
            names = [o["name"].replace(" ", "").lower() for o in objs]
            return names[:limit] if limit else names
        except Exception:
            return []

    @staticmethod
    def _extract_director(crew_str):
        try:
            crew = literal_eval(crew_str)
            return [m["name"].replace(" ", "").lower()
                    for m in crew if m.get("job") == "Director"]
        except Exception:
            return []

    def _build_tags(self, row):
        genres   = row["genres"]
        keywords = row["keywords"]
        cast     = row["cast"]
        crew     = row["crew"]
        overview = row["overview"].lower().split()
        tags     = genres + keywords + cast + crew + overview
        return " ".join(tags)

    # ── MODEL FITTING ────────────────────────────────────────────────────────

    def fit_content_model(self):
        df = self.movies_df
        df["tags"] = df.apply(self._build_tags, axis=1)

        cv = CountVectorizer(max_features=10_000, stop_words="english")
        vectors = cv.fit_transform(df["tags"])

        self.sim_matrix = cosine_similarity(vectors)
        self.indices     = pd.Series(df.index, index=df["title"]).drop_duplicates()
        print(f"Similarity matrix: {self.sim_matrix.shape}")
        return self

    def fit_svd_model(self, ratings_path=None):
        """
        Fit Surprise SVD on explicit ratings.
        Falls back to popularity-weighted pseudo-ratings if no ratings file.
        """
        try:
            from surprise import SVD, Dataset, Reader
            from surprise.model_selection import cross_validate

            if ratings_path and os.path.exists(ratings_path):
                ratings_df = pd.read_csv(ratings_path)
                reader = Reader(rating_scale=(0.5, 5.0))
                data   = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)
            else:
                # Build pseudo-ratings from TMDB vote data
                df = self.movies_df.copy()
                df = df[df["vote_count"] > 50].copy()
                df["pseudo_user"] = (df.index % 500).astype(str)
                df["rating"]      = (df["vote_average"] / 2).clip(0.5, 5.0)
                reader = Reader(rating_scale=(0.5, 5.0))
                data   = Dataset.load_from_df(df[["pseudo_user", "id", "rating"]], reader)

            self.svd_model = SVD(n_factors=50, n_epochs=20, random_state=42)
            cv_results     = cross_validate(self.svd_model, data, measures=["RMSE"], cv=3, verbose=False)
            print(f"SVD RMSE: {cv_results['test_rmse'].mean():.4f}")

            # Fit on full data
            trainset = data.build_full_trainset()
            self.svd_model.fit(trainset)

        except ImportError:
            print("Surprise not installed — collaborative model skipped. pip install scikit-surprise")

        return self

    # ── RECOMMENDATIONS ──────────────────────────────────────────────────────

    def get_content_recommendations(self, title, n=10):
        if title not in self.indices:
            return self._fuzzy_fallback(title, n)
        idx    = self.indices[title]
        scores = list(enumerate(self.sim_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
        results = []
        for i, score in scores:
            row = self.movies_df.iloc[i]
            results.append({
                "title":       row["title"],
                "year":        int(row["year"]),
                "genres":      row["genres"][:3],
                "similarity":  round(float(score), 4),
                "vote_avg":    round(float(row.get("vote_average", 0)), 1),
            })
        return results

    def get_collaborative_recommendations(self, user_id, n=10):
        if self.svd_model is None:
            return []
        # Predict ratings for all movies the user hasn't seen
        all_ids     = self.movies_df["id"].tolist()
        predictions = [self.svd_model.predict(str(user_id), mid) for mid in all_ids]
        predictions.sort(key=lambda x: x.est, reverse=True)
        results = []
        for pred in predictions[:n]:
            row = self.movies_df[self.movies_df["id"] == pred.iid]
            if row.empty:
                continue
            row = row.iloc[0]
            results.append({
                "title":      row["title"],
                "year":       int(row["year"]),
                "genres":     row["genres"][:3],
                "svd_score":  round(pred.est, 3),
            })
        return results

    def get_hybrid_recommendations(self, title, user_id=None, n=10,
                                   content_weight=0.55, collab_weight=0.45):
        content_recs = {r["title"]: r for r in self.get_content_recommendations(title, n*2)}
        all_titles   = list(content_recs.keys())

        if self.svd_model and user_id:
            collab_recs = {r["title"]: r for r in self.get_collaborative_recommendations(user_id, n*2)}
        else:
            collab_recs = {}

        hybrid = {}
        for t in set(all_titles) | set(collab_recs.keys()):
            c = content_recs.get(t, {}).get("similarity", 0)
            s = collab_recs.get(t,  {}).get("svd_score", 0) / 5.0
            hybrid[t] = c * content_weight + s * collab_weight

        top = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)[:n]
        results = []
        for t, score in top:
            src = content_recs.get(t) or collab_recs.get(t, {})
            src["hybrid_score"] = round(score, 4)
            results.append(src)
        return results

    def _fuzzy_fallback(self, title, n):
        """Simple substring matching when exact title not found."""
        q    = title.lower()
        mask = self.movies_df["title"].str.lower().str.contains(q, na=False)
        hits = self.movies_df[mask]
        if hits.empty:
            return []
        first = hits.iloc[0]["title"]
        return self.get_content_recommendations(first, n)

    # ── PERSISTENCE ──────────────────────────────────────────────────────────

    def save(self, path="backend/recommender.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved → {path}")

    @staticmethod
    def load(path="backend/recommender.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def build_and_save():
        rec = MovieRecommender()
        rec.load_data()
        rec.fit_content_model()
        rec.fit_svd_model()
        rec.save()
        return rec


if __name__ == "__main__":
    rec = MovieRecommender.build_and_save()
    print("\nContent recs for 'The Dark Knight':")
    for r in rec.get_content_recommendations("The Dark Knight"):
        print(f"  {r['title']} ({r['year']}) — {r['similarity']:.3f}")
