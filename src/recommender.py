import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import joblib

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"

AUDIO_FEATURES = ["danceability", "energy", "loudness", "speechiness",
                  "acousticness", "instrumentalness", "liveness",
                  "valence", "tempo", "duration_ms"]

class MusicRecommender:
    def __init__(self):
        self.df = pd.read_parquet(PROCESSED_DIR / "clustered.parquet")
        self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        self.feature_matrix = self.scaler.transform(self.df[AUDIO_FEATURES].values)
        self.has_year = "year" in self.df.columns

    def _result_cols(self, extra=None):
        base = ["track_name", "artists", "genre"]
        if self.has_year:
            base.append("year")
        if extra:
            base.extend(extra)
        return base

    def recommend_by_track(self, track_name, n=10, genre_filter=None,
                           decade_filter=None, popularity_weight=0.3):
        match = self.df[self.df["track_name"].str.lower() == track_name.lower()]
        if match.empty:
            match = self.df[self.df["track_name"].str.lower().str.contains(
                track_name.lower(), na=False)]
        if match.empty:
            raise ValueError(f"Track '{track_name}' not found in dataset")

        idx = match.index[0]
        query = self.feature_matrix[idx].reshape(1, -1)
        sim_scores = cosine_similarity(query, self.feature_matrix)[0]

        candidates = pd.DataFrame({
            "similarity": sim_scores,
            "popularity": self.df["popularity"].values,
            "genre": self.df["track_genre"].values,
            "decade": self.df["decade"].values,
            "track_name": self.df["track_name"].values,
            "artists": self.df["artists"].values,
        })
        if self.has_year:
            candidates["year"] = self.df["year"].values

        candidates = candidates[candidates.index != idx]

        if genre_filter and genre_filter != "All":
            candidates = candidates[candidates["genre"] == genre_filter]
        if decade_filter:
            candidates = candidates[candidates["decade"] == decade_filter]

        pop_min, pop_max = candidates["popularity"].min(), candidates["popularity"].max()
        if pop_max > pop_min:
            pop_norm = (candidates["popularity"] - pop_min) / (pop_max - pop_min)
        else:
            pop_norm = 0
        candidates["score"] = (1 - popularity_weight) * candidates["similarity"] \
                              + popularity_weight * pop_norm

        results = candidates.sort_values("score", ascending=False).head(n)
        return results[self._result_cols(["popularity", "similarity", "score"])]

    def recommend_by_features(self, features, n=10, genre_filter=None):
        query = np.array([features.get(f, 0) for f in AUDIO_FEATURES]).reshape(1, -1)
        query_scaled = self.scaler.transform(query)
        sim_scores = cosine_similarity(query_scaled, self.feature_matrix)[0]

        candidates = pd.DataFrame({
            "similarity": sim_scores,
            "popularity": self.df["popularity"].values,
            "genre": self.df["track_genre"].values,
            "track_name": self.df["track_name"].values,
            "artists": self.df["artists"].values,
        })
        if self.has_year:
            candidates["year"] = self.df["year"].values

        if genre_filter and genre_filter != "All":
            candidates = candidates[candidates["genre"] == genre_filter]

        pop_min, pop_max = candidates["popularity"].min(), candidates["popularity"].max()
        if pop_max > pop_min:
            pop_norm = (candidates["popularity"] - pop_min) / (pop_max - pop_min)
        else:
            pop_norm = 0
        candidates["score"] = 0.7 * candidates["similarity"] + 0.3 * pop_norm

        results = candidates.sort_values("score", ascending=False).head(n)
        return results[self._result_cols(["popularity", "similarity", "score"])]

    def recommend_by_genre(self, genre, n=10, min_popularity=30):
        pool = self.df[(self.df["track_genre"] == genre) &
                       (self.df["popularity"] >= min_popularity)]
        results = pool.sort_values("popularity", ascending=False).head(n)
        cols = ["track_name", "artists", "track_genre", "popularity"]
        if self.has_year:
            cols.append("year")
        return results[cols]

    def search_tracks(self, query, n=20):
        mask = self.df["track_name"].str.lower().str.contains(query.lower(), na=False)
        results = self.df[mask].sort_values("popularity", ascending=False).head(n)
        cols = ["track_name", "artists", "track_genre", "popularity"]
        if self.has_year:
            cols.append("year")
        return results[cols]
