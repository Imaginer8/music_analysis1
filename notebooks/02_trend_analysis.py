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

PROJECT_DIR = Path(__file__).resolve().parent.parent
df = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "features.parquet")
print(f"Data shape: {df.shape}")

# %% [markdown]
# ## 1. 音频特征随年代的演变

# %%
decade_features = ["danceability", "energy", "valence", "acousticness",
                   "instrumentalness", "speechiness", "loudness", "tempo"]
decade_avg = df.groupby("decade")[decade_features].mean().reset_index()
decade_avg = decade_avg[decade_avg["decade"] >= 1960]

fig = make_subplots(rows=2, cols=4, subplot_titles=decade_features)
for i, feat in enumerate(decade_features):
    row, col = i // 4 + 1, i % 4 + 1
    fig.add_trace(go.Scatter(x=decade_avg["decade"], y=decade_avg[feat],
                             mode="lines+markers", name=feat), row=row, col=col)
fig.update_layout(height=600, title_text="Audio Features Evolution Across Decades",
                  showlegend=False)
fig.write_html(PROJECT_DIR / "notebooks" / "feature_evolution.html")
print("Saved feature_evolution.html")

# %% [markdown]
# ## 2. 各年代流派热度的变化

# %%
genre_year = df.groupby(["decade", "track_genre"]).size().unstack(fill_value=0)
genre_year_pct = genre_year.div(genre_year.sum(axis=1), axis=0)
top_genres_over_time = genre_year_pct.sum().sort_values(ascending=False).head(8).index

fig, ax = plt.subplots(figsize=(14, 6))
for genre in top_genres_over_time:
    data = genre_year_pct[genre]
    data = data[data.index >= 1960]
    ax.plot(data.index, data.values, marker="o", label=genre, linewidth=2)
ax.set_title("Genre Share Over Decades (Top 8 Genres)")
ax.set_xlabel("Decade")
ax.set_ylabel("Percentage")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PROJECT_DIR / "notebooks" / "genre_share_trend.png", dpi=150)
print("Saved genre_share_trend.png")

# %% [markdown]
# ## 3. Explicit 内容趋势

# %%
explicit_trend = df.groupby("year")["explicit"].mean().reset_index()
explicit_trend = explicit_trend[explicit_trend["year"] >= 1970]

fig = px.line(explicit_trend, x="year", y="explicit",
              title="Explicit Content Ratio Over Time",
              labels={"explicit": "Explicit Ratio", "year": "Year"})
fig.add_hline(y=explicit_trend["explicit"].mean(), line_dash="dash",
              annotation_text=f"Overall avg: {explicit_trend['explicit'].mean():.3f}")
fig.write_html(PROJECT_DIR / "notebooks" / "explicit_trend.html")
print("Saved explicit_trend.html")

# %% [markdown]
# ## 4. 歌曲时长随年代变化

# %%
duration_trend = df.groupby("decade")["duration_min"].mean().reset_index()
duration_trend = duration_trend[duration_trend["decade"] >= 1960]

fig = px.bar(duration_trend, x="decade", y="duration_min",
             title="Average Song Duration by Decade",
             labels={"duration_min": "Duration (min)"},
             color="duration_min", color_continuous_scale="Viridis")
fig.write_html(PROJECT_DIR / "notebooks" / "duration_trend.html")
print("Saved duration_trend.html")

# %% [markdown]
# ## 5. 音频特征 vs 年代的回归分析（局部）

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plot_pairs = [("danceability", "energy"), ("valence", "acousticness"),
              ("year", "danceability"), ("year", "energy")]
for ax, (x, y) in zip(axes.flatten(), plot_pairs):
    if x == "year":
        sample = df.sample(min(5000, len(df)))
    else:
        sample = df.sample(min(5000, len(df)))
    sns.scatterplot(data=sample, x=x, y=y, alpha=0.3, ax=ax, s=10, color="steelblue")
    ax.set_title(f"{x} vs {y}")
plt.tight_layout()
plt.savefig(PROJECT_DIR / "notebooks" / "scatter_analysis.png", dpi=150)
print("Saved scatter_analysis.png")

# %% [markdown]
# ## 趋势总结

# %%
print("""
=== 趋势分析关键发现 ===
1. 近几十年 danceability/energy 稳步上升，acousticness 下降 → 音乐变得更"动感"
2. Speechiness 在 80s 后明显增长 → 说唱/Hip-Hop 影响扩大
3. Explicit 内容自 90s 起持续增加
4. 歌曲时长在 70s-90s 较长，近年趋短（注意力碎片化）
5. Pop/Hip-Hop 市占率上升，Rock/Classical 下降
""")
