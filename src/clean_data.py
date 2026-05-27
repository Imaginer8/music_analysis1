import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

def load_raw_data():
    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")
    df = pd.read_csv(csv_files[0])
    print(f"Loaded data: {df.shape}")
    return df

def clean_data(df):
    print("\n=== Data Cleaning ===")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        print("Dropped 'Unnamed: 0' index column")

    print(f"Duplicates: {df.duplicated().sum()}")
    df = df.drop_duplicates()

    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"Null values:\n{null_counts[null_counts > 0]}")
        df = df.dropna(subset=["artists", "track_name", "album_name"])

    num_cols = ["danceability", "energy", "loudness", "speechiness",
                "acousticness", "instrumentalness", "liveness", "valence",
                "tempo", "duration_ms", "popularity"]
    for col in num_cols:
        if col in df.columns:
            low = df[col].quantile(0.01)
            high = df[col].quantile(0.99)
            before = df.shape[0]
            df = df[(df[col] >= low) & (df[col] <= high)]
            print(f"{col}: clipped [{low:.3f}, {high:.3f}], removed {before - df.shape[0]} rows")

    if "explicit" in df.columns:
        df["explicit"] = df["explicit"].astype(int)

    df["duration_min"] = (df["duration_ms"] / 60000).round(2)

    if "year" in df.columns:
        df["decade"] = (df["year"] // 10) * 10
    else:
        df["decade"] = 0

    print(f"\nCleaned data shape: {df.shape}")
    return df

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw_data()
    df = clean_data(df)
    output_path = PROCESSED_DIR / "cleaned.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nSaved cleaned data to: {output_path}")

if __name__ == "__main__":
    main()
