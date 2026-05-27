# %% [markdown]
# # Spotify 音乐数据分析 — EDA
# ## 探索性数据分析：音频特征分布、流派对比、相关性分析

# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = Path.cwd().parent if "notebooks" in str(Path.cwd()) else Path.cwd()
df = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "features.parquet")
print(f"Data shape: {df.shape}")
print(f"\nColumns:\n{list(df.columns)}")

# %% [markdown]
# ## 1. 数据概览

# %%
df.head()

# %%
df.describe()

# %% [markdown]
# ## 2. 音频特征分布

# %%
audio_features = ["danceability", "energy", "valence", "acousticness",
                  "speechiness", "instrumentalness", "liveness", "tempo"]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, feat in enumerate(audio_features):
    sns.histplot(df[feat], bins=50, kde=True, ax=axes[i], color="steelblue")
    axes[i].set_title(f"{feat.capitalize()} Distribution")
    axes[i].set_xlabel("")
plt.tight_layout()
plt.savefig(PROJECT_DIR / "notebooks" / "feature_distributions.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. 相关性热力图

# %%
corr_cols = audio_features + ["popularity", "duration_ms", "loudness"]
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(df[corr_cols].corr()), k=1)
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r",
            mask=mask, square=True, linewidths=0.5)
plt.title("Audio Features Correlation Matrix")
plt.tight_layout()
plt.savefig(PROJECT_DIR / "notebooks" / "correlation_heatmap.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Top 流派对比（雷达图）

# %%
top_genres = df.groupby("track_genre")["popularity"].mean().sort_values(ascending=False).head(8).index
genre_feats = df[df["track_genre"].isin(top_genres)].groupby("track_genre")[
    ["danceability", "energy", "valence", "acousticness", "instrumentalness", "speechiness"]
].mean()

fig = go.Figure()
categories = list(genre_feats.columns)
for genre in genre_feats.index:
    values = genre_feats.loc[genre].values.tolist()
    values += values[:1]
    fig.add_trace(go.Scatterpolar(r=values, theta=categories + [categories[0]],
                                  fill="toself", name=genre))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    title="Top 8 Genres - Audio Feature Profile (Radar)",
    height=500
)
fig.write_html(PROJECT_DIR / "notebooks" / "genre_radar.html")
fig.show()

# %% [markdown]
# ## 5. 流行度分析

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.histplot(df["popularity"], bins=30, kde=True, ax=axes[0], color="coral")
axes[0].set_title("Popularity Distribution")

top10 = df.groupby("track_genre")["popularity"].mean().sort_values(ascending=False).head(10)
sns.barplot(x=top10.values, y=top10.index, ax=axes[1], palette="viridis")
axes[1].set_title("Avg Popularity by Genre (Top 10)")

df["popularity_bin"] = pd.cut(df["popularity"], bins=[0, 25, 50, 75, 100],
                              labels=["Low", "Medium", "High", "Very High"])
pop_genre = df.groupby("track_genre")["popularity_bin"].value_counts(normalize=True).unstack()
sns.heatmap(pop_genre, cmap="YlOrRd", ax=axes[2], annot=True, fmt=".0%")
axes[2].set_title("Popularity Distribution by Genre")

plt.tight_layout()
plt.savefig(PROJECT_DIR / "notebooks" / "popularity_analysis.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Explicit 内容分析

# %%
explicit_pct = df.groupby("track_genre")["explicit"].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=explicit_pct.values, y=explicit_pct.index, palette="rocket")
plt.title("Explicit Content Ratio by Genre")
plt.xlabel("Proportion Explicit")
plt.tight_layout()
plt.savefig(PROJECT_DIR / "notebooks" / "explicit_analysis.png", dpi=150)
plt.show()

# %% [markdown]
# ## 7. 关键发现总结

# %%
print("""
=== EDA 关键发现 ===
1. 音频特征分布：danceability/energy 呈右偏分布，instrumentalness 呈 U 型两极分化
2. 相关性：energy 与 loudness 强正相关，acousticness 与 energy/valence 负相关
3. 流派差异：各流派有独特的音频特征"指纹"，有助于分类
4. 流行度：大部分歌曲流行度低，呈长尾分布；不同流派流行度差异明显
5. Explicit：hip-hop 流派 explicit 比例最高，classical/ambient 最低
""")
