import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
np.random.seed(42)

GENRES = [
    "pop", "rock", "hip-hop", "r&b", "electronic", "jazz", "classical",
    "country", "reggae", "blues", "metal", "indie", "folk", "latin",
    "soul", "punk", "alternative", "edm", "k-pop", "ambient"
]

def generate_song_name():
    prefixes = ["Midnight", "Golden", "Crimson", "Electric", "Velvet", "Broken",
                "Neon", "Silent", "Burning", "Fading", "Cosmic", "Crystal",
                "Thunder", "Whisper", "Ocean", "Shadow", "Wild", "Sacred",
                "Hollow", "Rising", "Fallen", "Dancing", "Lost", "Bitter"]
    suffixes = ["Dreams", "Fire", "Heart", "Light", "Night", "Shadows",
                "Tears", "Storm", "Memory", "River", "Grace", "Echo",
                "Waves", "Rain", "Sky", "Road", "Stars", "Moon",
                "Sunrise", "Silence", "Desire", "Fury", "Hope", "Pain"]
    return f"{np.random.choice(prefixes)} {np.random.choice(suffixes)}"

def generate_artist_name():
    prefixes = ["The", "Lunar", "Crimson", "Electric", "Velvet", "Broken", "Neon"]
    cores = ["Echoes", "Waves", "Shadows", "Dreamers", "Riders", "Soul", "Kings",
             "Makers", "Rebels", "Vipers", "Tides", "Phoenix", "Thunder"]
    return f"{np.random.choice(prefixes)} {np.random.choice(cores)}"

def generate_album_name():
    templates = [
        "{} Nights", "The {} Chronicles", "{} Boulevard",
        "Songs of {}", "{} Avenue", "Heart of {}", "{} Diaries"
    ]
    words = ["Electric", "Neon", "Crimson", "Faded", "Golden", "Velvet",
             "Empty", "Midnight", "Broken", "Cosmic", "Silent", "Wicked"]
    t = np.random.choice(templates)
    return t.format(np.random.choice(words))

genre_params = {
    "pop":       {"danceability": (0.65, 0.12), "energy": (0.70, 0.13), "valence": (0.55, 0.15), "acousticness": (0.20, 0.15), "instrumentalness": (0.05, 0.08), "speechiness": (0.08, 0.05), "liveness": (0.18, 0.10), "loudness": (-5.0, 2.5), "tempo": (118, 14), "duration_s": (210, 30)},
    "rock":      {"danceability": (0.45, 0.13), "energy": (0.80, 0.10), "valence": (0.50, 0.16), "acousticness": (0.15, 0.12), "instrumentalness": (0.10, 0.12), "speechiness": (0.05, 0.03), "liveness": (0.20, 0.12), "loudness": (-4.0, 2.0), "tempo": (130, 18), "duration_s": (240, 40)},
    "hip-hop":   {"danceability": (0.75, 0.10), "energy": (0.65, 0.14), "valence": (0.40, 0.18), "acousticness": (0.10, 0.10), "instrumentalness": (0.05, 0.08), "speechiness": (0.25, 0.12), "liveness": (0.15, 0.10), "loudness": (-3.0, 2.0), "tempo": (95, 12), "duration_s": (210, 35)},
    "r&b":       {"danceability": (0.60, 0.12), "energy": (0.55, 0.15), "valence": (0.45, 0.16), "acousticness": (0.25, 0.13), "instrumentalness": (0.03, 0.05), "speechiness": (0.12, 0.06), "liveness": (0.15, 0.09), "loudness": (-6.0, 2.5), "tempo": (105, 15), "duration_s": (225, 35)},
    "electronic":{"danceability": (0.70, 0.11), "energy": (0.78, 0.12), "valence": (0.45, 0.18), "acousticness": (0.08, 0.10), "instrumentalness": (0.40, 0.25), "speechiness": (0.05, 0.04), "liveness": (0.12, 0.08), "loudness": (-4.5, 2.5), "tempo": (125, 16), "duration_s": (260, 50)},
    "jazz":      {"danceability": (0.40, 0.14), "energy": (0.30, 0.15), "valence": (0.40, 0.18), "acousticness": (0.75, 0.15), "instrumentalness": (0.60, 0.25), "speechiness": (0.04, 0.03), "liveness": (0.20, 0.12), "loudness": (-10.0, 3.0), "tempo": (100, 25), "duration_s": (280, 60)},
    "classical": {"danceability": (0.20, 0.10), "energy": (0.25, 0.14), "valence": (0.30, 0.15), "acousticness": (0.90, 0.08), "instrumentalness": (0.85, 0.12), "speechiness": (0.03, 0.02), "liveness": (0.25, 0.12), "loudness": (-15.0, 4.0), "tempo": (110, 30), "duration_s": (350, 100)},
    "country":   {"danceability": (0.50, 0.12), "energy": (0.60, 0.14), "valence": (0.55, 0.16), "acousticness": (0.50, 0.18), "instrumentalness": (0.05, 0.08), "speechiness": (0.05, 0.03), "liveness": (0.18, 0.10), "loudness": (-6.0, 2.5), "tempo": (120, 20), "duration_s": (220, 35)},
    "reggae":    {"danceability": (0.65, 0.10), "energy": (0.50, 0.14), "valence": (0.55, 0.14), "acousticness": (0.25, 0.12), "instrumentalness": (0.15, 0.15), "speechiness": (0.10, 0.06), "liveness": (0.20, 0.12), "loudness": (-5.0, 2.0), "tempo": (85, 12), "duration_s": (240, 40)},
    "blues":     {"danceability": (0.45, 0.12), "energy": (0.45, 0.16), "valence": (0.40, 0.17), "acousticness": (0.55, 0.18), "instrumentalness": (0.25, 0.20), "speechiness": (0.05, 0.03), "liveness": (0.22, 0.12), "loudness": (-7.0, 3.0), "tempo": (105, 20), "duration_s": (260, 50)},
    "metal":     {"danceability": (0.35, 0.12), "energy": (0.90, 0.08), "valence": (0.35, 0.16), "acousticness": (0.05, 0.06), "instrumentalness": (0.15, 0.15), "speechiness": (0.08, 0.05), "liveness": (0.25, 0.14), "loudness": (-2.5, 1.5), "tempo": (140, 25), "duration_s": (260, 50)},
    "indie":     {"danceability": (0.50, 0.14), "energy": (0.55, 0.16), "valence": (0.45, 0.18), "acousticness": (0.40, 0.20), "instrumentalness": (0.15, 0.16), "speechiness": (0.06, 0.04), "liveness": (0.18, 0.10), "loudness": (-7.0, 3.0), "tempo": (120, 22), "duration_s": (240, 45)},
    "folk":      {"danceability": (0.40, 0.13), "energy": (0.40, 0.16), "valence": (0.45, 0.18), "acousticness": (0.70, 0.16), "instrumentalness": (0.15, 0.15), "speechiness": (0.04, 0.03), "liveness": (0.20, 0.12), "loudness": (-8.0, 3.0), "tempo": (110, 22), "duration_s": (230, 40)},
    "latin":     {"danceability": (0.75, 0.10), "energy": (0.70, 0.12), "valence": (0.60, 0.14), "acousticness": (0.15, 0.12), "instrumentalness": (0.03, 0.05), "speechiness": (0.10, 0.06), "liveness": (0.15, 0.10), "loudness": (-4.0, 2.0), "tempo": (110, 20), "duration_s": (210, 30)},
    "soul":      {"danceability": (0.55, 0.12), "energy": (0.55, 0.14), "valence": (0.50, 0.16), "acousticness": (0.35, 0.16), "instrumentalness": (0.05, 0.08), "speechiness": (0.08, 0.05), "liveness": (0.18, 0.10), "loudness": (-5.5, 2.5), "tempo": (100, 18), "duration_s": (230, 40)},
    "punk":      {"danceability": (0.45, 0.12), "energy": (0.85, 0.10), "valence": (0.45, 0.16), "acousticness": (0.08, 0.08), "instrumentalness": (0.08, 0.10), "speechiness": (0.06, 0.04), "liveness": (0.25, 0.14), "loudness": (-3.0, 1.8), "tempo": (155, 25), "duration_s": (190, 35)},
    "alternative":{"danceability": (0.50, 0.14), "energy": (0.65, 0.15), "valence": (0.40, 0.18), "acousticness": (0.25, 0.18), "instrumentalness": (0.20, 0.20), "speechiness": (0.06, 0.04), "liveness": (0.18, 0.11), "loudness": (-6.0, 3.0), "tempo": (125, 24), "duration_s": (250, 45)},
    "edm":       {"danceability": (0.72, 0.10), "energy": (0.85, 0.10), "valence": (0.45, 0.18), "acousticness": (0.05, 0.06), "instrumentalness": (0.35, 0.25), "speechiness": (0.04, 0.03), "liveness": (0.12, 0.08), "loudness": (-3.5, 2.0), "tempo": (128, 10), "duration_s": (240, 45)},
    "k-pop":     {"danceability": (0.70, 0.10), "energy": (0.78, 0.11), "valence": (0.50, 0.16), "acousticness": (0.12, 0.10), "instrumentalness": (0.02, 0.04), "speechiness": (0.10, 0.06), "liveness": (0.14, 0.09), "loudness": (-4.0, 2.0), "tempo": (115, 18), "duration_s": (210, 25)},
    "ambient":   {"danceability": (0.15, 0.10), "energy": (0.20, 0.12), "valence": (0.25, 0.14), "acousticness": (0.60, 0.20), "instrumentalness": (0.80, 0.15), "speechiness": (0.03, 0.02), "liveness": (0.12, 0.08), "loudness": (-14.0, 4.0), "tempo": (90, 25), "duration_s": (350, 100)},
}

KEYS = list(range(12))
MODES = [0, 1]
TIME_SIGNATURES = [3, 4, 5, 6, 7]

def clip(val, low=0.0, high=1.0):
    return max(low, min(high, val))

def generate_track(year, genre):
    params = genre_params[genre]
    danceability = clip(np.random.normal(*params["danceability"]))
    energy = clip(np.random.normal(*params["energy"]))
    valence = clip(np.random.normal(*params["valence"]))
    acousticness = clip(np.random.normal(*params["acousticness"]))
    instrumentalness = clip(np.random.normal(*params["instrumentalness"]))
    speechiness = clip(np.random.normal(*params["speechiness"]))
    liveness = clip(np.random.normal(*params["liveness"]))
    loudness = clip(np.random.normal(*params["loudness"]), -30, 0)
    tempo = clip(np.random.normal(*params["tempo"]), 30, 220)
    duration_ms = int(clip(np.random.normal(*params["duration_s"]), 60, 600) * 1000)

    popularity = np.random.randint(0, 101)
    explicit = int(np.random.random() < 0.15)
    key = int(np.random.choice(KEYS))
    mode = int(np.random.choice(MODES))
    time_signature = int(np.random.choice(TIME_SIGNATURES))

    track_id = f"spotify:track:{np.random.bytes(8).hex()}"
    artists = ", ".join([generate_artist_name() for _ in range(np.random.randint(1, 4))])
    track_name = generate_song_name()
    album_name = generate_album_name()

    return {
        "track_id": track_id,
        "artists": artists,
        "album_name": album_name,
        "track_name": track_name,
        "popularity": popularity,
        "duration_ms": duration_ms,
        "explicit": explicit,
        "danceability": round(danceability, 4),
        "energy": round(energy, 4),
        "key": key,
        "loudness": round(loudness, 2),
        "mode": mode,
        "speechiness": round(speechiness, 4),
        "acousticness": round(acousticness, 4),
        "instrumentalness": round(instrumentalness, 4),
        "liveness": round(liveness, 4),
        "valence": round(valence, 4),
        "tempo": round(tempo, 2),
        "time_signature": time_signature,
        "track_genre": genre,
        "year": year,
    }

def generate_dataset(n_tracks=114000):
    years = list(range(1950, 2026))
    year_weights = [0.5**abs(2020 - y) + 0.1 for y in years]
    year_weights = [w / sum(year_weights) for w in year_weights]

    genre_counts = {
        "pop": 0.14, "rock": 0.12, "hip-hop": 0.10, "r&b": 0.07,
        "electronic": 0.06, "jazz": 0.05, "classical": 0.04, "country": 0.05,
        "reggae": 0.03, "blues": 0.03, "metal": 0.04, "indie": 0.05,
        "folk": 0.03, "latin": 0.04, "soul": 0.03, "punk": 0.03,
        "alternative": 0.04, "edm": 0.03, "k-pop": 0.02, "ambient": 0.02,
    }
    genre_list = []
    for g, w in genre_counts.items():
        genre_list.extend([g] * max(1, int(n_tracks * w)))
    while len(genre_list) < n_tracks:
        genre_list.append("pop")
    np.random.shuffle(genre_list)
    genre_list = genre_list[:n_tracks]

    records = []
    for i in range(n_tracks):
        year = int(np.random.choice(years, p=year_weights))
        genre = genre_list[i]
        records.append(generate_track(year, genre))
        if (i + 1) % 20000 == 0:
            print(f"Generated {i + 1}/{n_tracks} tracks...")

    return pd.DataFrame(records)

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating Spotify-like dataset...")
    df = generate_dataset(114000)
    output_path = RAW_DIR / "spotify_tracks_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    main()
