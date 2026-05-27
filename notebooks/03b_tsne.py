from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
df = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "clustered.parquet")
print(f"Data shape: {df.shape}")

CLUSTER_FEATURES = ["danceability", "energy", "valence", "acousticness",
                    "instrumentalness", "speechiness", "liveness", "tempo"]

sample = df.sample(5000, random_state=42)
X_sample = sample[CLUSTER_FEATURES].values

print("Running t-SNE on 5000 samples...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500, learning_rate=200)
X_tsne = tsne.fit_transform(X_sample)

sample["tsne1"] = X_tsne[:, 0]
sample["tsne2"] = X_tsne[:, 1]

fig = px.scatter(sample, x="tsne1", y="tsne2",
                 color=sample["cluster"].astype(str),
                 title="t-SNE Visualization of Clusters (5000 samples)",
                 labels={"color": "Cluster"},
                 opacity=0.6, width=800, height=600)
fig.write_html(PROJECT_DIR / "notebooks" / "tsne_clusters.html")
print("Saved tsne_clusters.html")
