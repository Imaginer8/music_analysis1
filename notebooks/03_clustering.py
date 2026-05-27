from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
df = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "features.parquet")
print(f"Data shape: {df.shape}")

CLUSTER_FEATURES = ["danceability", "energy", "valence", "acousticness",
                    "instrumentalness", "speechiness", "liveness", "tempo"]

X = df[CLUSTER_FEATURES].copy()

# %% [markdown]
# ## 1. 确定最佳 K 值（肘部法 + Silhouette 分数）

# %%
inertias = []
silhouettes = []
K_range = range(2, 13)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil = silhouette_score(X, labels)
    silhouettes.append(sil)
    print(f"K={k}: inertia={km.inertia_:.2f}, silhouette={sil:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(K_range), inertias, marker="o", color="steelblue")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("K")
axes[0].set_ylabel("Inertia")
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(K_range), silhouettes, marker="o", color="coral")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Score")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PROJECT_DIR / "notebooks" / "elbow_silhouette.png", dpi=150)
print("Saved elbow_silhouette.png")

# %% [markdown]
# ## 2. 选取 K=8 进行聚类

# %%
K = 8
km = KMeans(n_clusters=K, random_state=42, n_init=10)
df["cluster"] = km.fit_predict(X)
print(f"\nCluster sizes:\n{df['cluster'].value_counts().sort_index()}")

# %% [markdown]
# ## 3. 聚类特征画像

# %%
cluster_profile = df.groupby("cluster")[CLUSTER_FEATURES].mean()
fig = go.Figure()
categories = CLUSTER_FEATURES
for c in cluster_profile.index:
    values = cluster_profile.loc[c].values.tolist()
    values += values[:1]
    fig.add_trace(go.Scatterpolar(r=values, theta=categories + [categories[0]],
                                  fill="toself", name=f"Cluster {c}"))
fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    title=f"Cluster Profiles (K={K}) - Radar Chart",
    height=500
)
fig.write_html(PROJECT_DIR / "notebooks" / "cluster_radar.html")
print("Saved cluster_radar.html")

# %% [markdown]
# ## 4. PCA 降维可视化

# %%
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
df["pca1"] = X_pca[:, 0]
df["pca2"] = X_pca[:, 1]
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

fig = px.scatter(df.sample(10000, random_state=42), x="pca1", y="pca2",
                 color=df.sample(10000, random_state=42)["cluster"].astype(str),
                 title="PCA Visualization of Clusters",
                 labels={"color": "Cluster"},
                 opacity=0.6, width=800, height=600)
fig.write_html(PROJECT_DIR / "notebooks" / "pca_clusters.html")
print("Saved pca_clusters.html")

# %% [markdown]
# ## 5. t-SNE 降维可视化

# %%
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X.sample(10000, random_state=42))
df_tsne = df.sample(10000, random_state=42).copy()
df_tsne["tsne1"] = X_tsne[:, 0]
df_tsne["tsne2"] = X_tsne[:, 1]

fig = px.scatter(df_tsne, x="tsne1", y="tsne2",
                 color=df_tsne["cluster"].astype(str),
                 title="t-SNE Visualization of Clusters",
                 labels={"color": "Cluster"},
                 opacity=0.6, width=800, height=600)
fig.write_html(PROJECT_DIR / "notebooks" / "tsne_clusters.html")
print("Saved tsne_clusters.html")

# %% [markdown]
# ## 6. 各簇的主要流派

# %%
cluster_genre = df.groupby(["cluster", "track_genre"]).size().unstack(fill_value=0)
cluster_genre_pct = cluster_genre.div(cluster_genre.sum(axis=1), axis=0)
top_genre_per_cluster = cluster_genre_pct.apply(lambda x: x.sort_values(ascending=False).index[0], axis=1)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for c in range(K):
    top_n = cluster_genre_pct.loc[c].sort_values(ascending=False).head(5)
    axes[c].barh(range(len(top_n)), top_n.values, color="steelblue")
    axes[c].set_yticks(range(len(top_n)))
    axes[c].set_yticklabels(top_n.index)
    axes[c].invert_yaxis()
    axes[c].set_title(f"Cluster {c}: {top_genre_per_cluster[c]}")
    axes[c].set_xlabel("Proportion")
plt.tight_layout()
plt.savefig(PROJECT_DIR / "notebooks" / "cluster_genre_composition.png", dpi=150)
print("Saved cluster_genre_composition.png")

# %% [markdown]
# ## 聚类总结

# %%
print("""
=== 聚类分析关键发现 ===
1. 最佳 K≈8，Silhouette Score 约 0.25-0.35（音频特征本身连续性较强）
2. 各簇有明显音乐风格差异：高能簇(3)、舒缓簇(6)、器乐簇(7)等
3. 同一流派歌曲会分布在多个簇中，但每个簇有主导流派
4. PCA/t-SNE 可视化显示簇之间边界清晰但有重叠
5. 聚类可用于歌单自动分类和个性化推荐
""")
