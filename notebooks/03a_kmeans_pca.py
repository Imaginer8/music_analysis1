from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
df = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "features.parquet")
print(f"Data shape: {df.shape}")

CLUSTER_FEATURES = ["danceability", "energy", "valence", "acousticness",
                    "instrumentalness", "speechiness", "liveness", "tempo"]
X = df[CLUSTER_FEATURES].copy()

sample_elbow = X.sample(20000, random_state=42)
K_range = range(2, 13)
inertias, silhouettes = [], []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=5)
    labels = km.fit_predict(sample_elbow)
    inertias.append(km.inertia_)
    sil = silhouette_score(sample_elbow, labels)
    silhouettes.append(sil)
    print(f"K={k}: inertia={km.inertia_:.2f}, silhouette={sil:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(K_range), inertias, marker="o", color="steelblue")
axes[0].set_title("Elbow Method"); axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")
axes[1].plot(list(K_range), silhouettes, marker="o", color="coral")
axes[1].set_title("Silhouette Score"); axes[1].set_xlabel("K"); axes[1].set_ylabel("Score")
plt.tight_layout()
plt.savefig(PROJECT_DIR / "notebooks" / "elbow_silhouette.png", dpi=150)
print("Saved elbow_silhouette.png")

K = 8
km = KMeans(n_clusters=K, random_state=42, n_init=5)
df["cluster"] = km.fit_predict(X)
print(f"Cluster sizes:\n{df['cluster'].value_counts().sort_index()}")

cluster_profile = df.groupby("cluster")[CLUSTER_FEATURES].mean()
fig = go.Figure()
categories = CLUSTER_FEATURES
for c in cluster_profile.index:
    values = cluster_profile.loc[c].values.tolist() + [cluster_profile.loc[c].values[0]]
    fig.add_trace(go.Scatterpolar(r=values, theta=categories + [categories[0]],
                                  fill="toself", name=f"Cluster {c}"))
fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                  title="Cluster Profiles (K=8)", height=500)
fig.write_html(PROJECT_DIR / "notebooks" / "cluster_radar.html")
print("Saved cluster_radar.html")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
df["pca1"], df["pca2"] = X_pca[:, 0], X_pca[:, 1]
print(f"PCA variance ratio: {pca.explained_variance_ratio_}")

fig = px.scatter(df.sample(10000, random_state=42), x="pca1", y="pca2",
                 color=df.sample(10000, random_state=42)["cluster"].astype(str),
                 title="PCA Cluster Visualization", opacity=0.6, width=800, height=600)
fig.write_html(PROJECT_DIR / "notebooks" / "pca_clusters.html")
print("Saved pca_clusters.html")

cluster_genre = df.groupby(["cluster", "track_genre"]).size().unstack(fill_value=0)
cluster_genre_pct = cluster_genre.div(cluster_genre.sum(axis=1), axis=0)
top_genre_per_cluster = cluster_genre_pct.apply(
    lambda x: x.sort_values(ascending=False).index[0], axis=1)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for c in range(K):
    top_n = cluster_genre_pct.loc[c].sort_values(ascending=False).head(5)
    axes[c].barh(range(len(top_n)), top_n.values, color="steelblue")
    axes[c].set_yticks(range(len(top_n)))
    axes[c].set_yticklabels(top_n.index)
    axes[c].invert_yaxis()
    axes[c].set_title(f"Cluster {c}: {top_genre_per_cluster[c]}")
plt.tight_layout()
plt.savefig(PROJECT_DIR / "notebooks" / "cluster_genre_composition.png", dpi=150)
print("Saved cluster_genre_composition.png")

output = PROJECT_DIR / "data" / "processed" / "clustered.parquet"
df.to_parquet(output, index=False)
print(f"Saved clustered data to {output}")
print("Done!")
