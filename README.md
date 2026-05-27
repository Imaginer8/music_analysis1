# Spotify Music Data Analysis & Recommendation System

An end-to-end data analysis and recommendation system built on 114,000 Spotify tracks across 20 genres.

## Features

- **Exploratory Data Analysis**: Audio feature distributions, genre comparisons, correlation analysis
- **Music Trend Analysis**: Decades-long evolution of audio features, genre popularity shifts, explicit content trends
- **Clustering**: KMeans clustering (K=8) with PCA/t-SNE visualization to discover music style groups
- **Hybrid Recommendation System**: Content-based (cosine similarity) + popularity-weighted + genre-filtered recommendations
- **Interactive Dashboard**: Streamlit web app with 5 pages for visual exploration

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn, Plotly |
| Clustering | KMeans, PCA, t-SNE |
| Recommendation | Cosine Similarity, Hybrid Scoring |
| Web App | Streamlit |

## Project Structure

```
spotify-analysis/
├── data/
│   ├── raw/                    # Raw dataset
│   └── processed/              # Cleaned & feature-engineered data
├── notebooks/                  # Analysis scripts
│   ├── 01_eda.py               # Exploratory data analysis
│   ├── 02_trend_analysis.py    # Music trend analysis
│   ├── 03a_kmeans_pca.py       # KMeans clustering & PCA
│   ├── 03b_tsne.py             # t-SNE visualization
│   └── 04_recommendation.py    # Recommendation demo
├── src/
│   ├── clean_data.py           # Data cleaning pipeline
│   ├── features.py             # Feature engineering
│   ├── recommender.py          # Recommendation engine
│   └── generate_data.py        # Synthetic data generation
├── models/                     # Saved encoders & scalers
├── app.py                      # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Quick Start

1. Clone the repository
   ```bash
   git clone https://github.com/Imaginer8/music_analysis1.git
   cd music_analysis1
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run analysis notebooks
   ```bash
   python notebooks/01_eda.py
   python notebooks/02_trend_analysis.py
   python notebooks/03a_kmeans_pca.py
   ```

4. Launch the Streamlit dashboard
   ```bash
   streamlit run app.py
   ```

## Key Findings

- **Music is becoming more energetic**: Danceability and energy have steadily increased since the 1960s, while acousticness has declined
- **Genre fingerprints**: Each genre has a distinct audio feature profile that enables accurate classification
- **8 style clusters**: Unsupervised clustering reveals 8 distinct music style groups with clear genre affinities
- **Explicit content on the rise**: The proportion of explicit songs has grown significantly since the 1990s
- **Shorter songs trend**: Average song duration has decreased in recent decades
