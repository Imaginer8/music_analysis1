import kagglehub
import shutil
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

def download_spotify_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading Spotify Tracks Dataset from Kaggle...")
    path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")

    src_path = Path(path)
    for f in src_path.glob("*.csv"):
        dest = RAW_DIR / f.name
        shutil.copy2(str(f), str(dest))
        print(f"Copied: {f.name} -> {dest}")

    print(f"Dataset downloaded to: {RAW_DIR}")

if __name__ == "__main__":
    download_spotify_dataset()
