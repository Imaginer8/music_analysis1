import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from pathlib import Path
import joblib

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"

AUDIO_FEATURES = ["danceability", "energy", "loudness", "speechiness",
                  "acousticness", "instrumentalness", "liveness",
                  "valence", "tempo", "duration_ms"]

def load_cleaned():
    path = PROCESSED_DIR / "cleaned.parquet"
    return pd.read_parquet(path)

def encode_genre(df):
    le = LabelEncoder()
    df["genre_encoded"] = le.fit_transform(df["track_genre"])
    return df, le

def scale_features(df):
    scaler = MinMaxScaler()
    df[AUDIO_FEATURES] = scaler.fit_transform(df[AUDIO_FEATURES])
    return df, scaler

def extract_year_features(df):
    df["release_quarter"] = df["year"] % 4

    if "released_month" in df.columns:
        df["season"] = df["released_month"].map({12: 0, 1: 0, 2: 0,
                                                  3: 1, 4: 1, 5: 1,
                                                  6: 2, 7: 2, 8: 2,
                                                  9: 3, 10: 3, 11: 3})

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_cleaned()
    print(f"Loaded: {df.shape}")

    df, genre_encoder = encode_genre(df)
    df, scaler = scale_features(df)

    joblib.dump(genre_encoder, MODELS_DIR / "genre_encoder.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print(f"Saved encoder & scaler to {MODELS_DIR}")

    output = PROCESSED_DIR / "features.parquet"
    df.to_parquet(output, index=False)
    print(f"Saved feature data to: {output}")

if __name__ == "__main__":
    main()
