from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.recommender import MusicRecommender

rec = MusicRecommender()
print(f"Loaded {len(rec.df)} tracks")

sample_track = rec.df["track_name"].iloc[0]
print(f"\nInput track: '{sample_track}' by {rec.df['artists'].iloc[0]}")

results = rec.recommend_by_track(sample_track, n=5)
print("\n=== Content-based Recommendations ===")
print(results.to_string(index=False))

results_pop = rec.recommend_by_track(sample_track, n=5, popularity_weight=0.5)
print("\n=== Hybrid (Popularity-weighted) Recommendations ===")
print(results_pop.to_string(index=False))

genre_results = rec.recommend_by_genre("pop", n=5, min_popularity=50)
print("\n=== Top Pop Songs (popularity >= 50) ===")
print(genre_results.to_string(index=False))

search_results = rec.search_tracks("dream", n=5)
print("\n=== Search 'dream' ===")
print(search_results.to_string(index=False))

feature_results = rec.recommend_by_features(
    {"danceability": 0.8, "energy": 0.7, "valence": 0.6, "tempo": 120}, n=5
)
print("\n=== Feature-based Recommendations (danceable, energetic) ===")
print(feature_results.to_string(index=False))
