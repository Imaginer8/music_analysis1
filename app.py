import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.recommender import MusicRecommender, AUDIO_FEATURES

CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR / "data" / "processed"

st.set_page_config(page_title="Spotify Music Analysis", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_DIR / "clustered.parquet")
    return df

@st.cache_resource
def load_recommender():
    return MusicRecommender()

df = load_data()
rec = load_recommender()
has_year = "year" in df.columns

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "EDA", "Trends", "Clusters", "Recommender"])

if page == "Overview":
    st.title("Spotify Music Data Analysis")
    st.markdown("A comprehensive analysis of 114,000 Spotify tracks across 20 genres, including audio feature exploration, trend analysis, clustering, and a hybrid recommendation system.")

    display_cols = ["track_name", "artists", "track_genre", "popularity"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tracks", f"{len(df):,}")
    col2.metric("Genres", df["track_genre"].nunique())
    col3.metric("Artists", df["artists"].nunique())
    col4.metric("Avg Popularity", f"{df['popularity'].mean():.1f}")

    st.subheader("Data Sample")
    st.dataframe(df[display_cols].head(10), width='stretch')

    st.subheader("Genre Distribution")
    genre_counts = df["track_genre"].value_counts()
    fig = px.bar(genre_counts, x=genre_counts.index, y=genre_counts.values,
                 title="Track Count by Genre", color=genre_counts.values,
                 color_continuous_scale="Viridis")
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, width='stretch')

elif page == "EDA":
    st.title("Exploratory Data Analysis")

    feat = st.selectbox("Select Audio Feature", AUDIO_FEATURES, index=0)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x=feat, nbins=50, title=f"{feat.capitalize()} Distribution",
                           color_discrete_sequence=["steelblue"])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

    with col2:
        fig = px.box(df, x="track_genre", y=feat,
                     title=f"{feat.capitalize()} by Genre")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, width='stretch')

    st.subheader("Correlation Matrix")
    corr_cols = AUDIO_FEATURES + ["popularity"]
    corr = df[corr_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    title="Feature Correlation", aspect="auto", height=600)
    st.plotly_chart(fig, width='stretch')

    st.subheader("Popularity Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="popularity", nbins=30,
                           title="Popularity Distribution", color_discrete_sequence=["coral"])
        st.plotly_chart(fig, width='stretch')
    with col2:
        top_genres = df.groupby("track_genre")["popularity"].mean().sort_values(ascending=False).head(15)
        fig = px.bar(top_genres, x=top_genres.values, y=top_genres.index,
                     orientation="h", title="Avg Popularity by Genre",
                     color=top_genres.values, color_continuous_scale="Viridis")
        st.plotly_chart(fig, width='stretch')

    st.subheader("Genre Radar")
    top8 = df.groupby("track_genre")["popularity"].mean().sort_values(ascending=False).head(8).index
    genre_feats = df[df["track_genre"].isin(top8)].groupby("track_genre")[
        AUDIO_FEATURES[:6]].mean()
    fig = go.Figure()
    for genre in genre_feats.index:
        values = genre_feats.loc[genre].values.tolist()
        fig.add_trace(go.Scatterpolar(r=values + [values[0]],
                                      theta=list(genre_feats.columns) + [genre_feats.columns[0]],
                                      fill="toself", name=genre))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=500)
    st.plotly_chart(fig, width='stretch')

elif page == "Trends":
    st.title("Music Trend Analysis")

    if has_year:
        decades = sorted(df["decade"].unique())
        decade_range = st.slider("Select Decade Range", int(decades[0]), int(decades[-1]),
                                 (int(decades[0]), int(decades[-1])))
        mask = (df["decade"] >= decade_range[0]) & (df["decade"] <= decade_range[1])
        df_filtered = df[mask]

        trend_feat = st.multiselect("Features to Show",
                                    AUDIO_FEATURES, default=["danceability", "energy", "valence"])

        if trend_feat:
            decade_avg = df_filtered.groupby("decade")[trend_feat].mean().reset_index()
            fig = go.Figure()
            for feat in trend_feat:
                fig.add_trace(go.Scatter(x=decade_avg["decade"], y=decade_avg[feat],
                                         mode="lines+markers", name=feat))
            fig.update_layout(title="Audio Features Over Decades", xaxis_title="Decade",
                              yaxis_title="Average Value", height=500)
            st.plotly_chart(fig, width='stretch')

        st.subheader("Genre Share Over Time")
        genre_year = df_filtered.groupby(["decade", "track_genre"]).size().unstack(fill_value=0)
        genre_pct = genre_year.div(genre_year.sum(axis=1), axis=0)
        top_genres = genre_pct.sum().sort_values(ascending=False).head(8).index
        genre_pct_top = genre_pct[top_genres]

        fig = go.Figure()
        for genre in top_genres:
            fig.add_trace(go.Scatter(x=genre_pct_top.index, y=genre_pct_top[genre],
                                     mode="lines+markers", name=genre, stackgroup="one"))
        fig.update_layout(title="Genre Share Over Decades", xaxis_title="Decade",
                          yaxis_title="Share", height=500)
        st.plotly_chart(fig, width='stretch')

        st.subheader("Explicit Content Trend")
        explicit_trend = df_filtered.groupby("year")["explicit"].mean().reset_index()
        fig = px.line(explicit_trend, x="year", y="explicit",
                      title="Explicit Content Ratio Over Years",
                      labels={"explicit": "Ratio"})
        fig.add_hline(y=explicit_trend["explicit"].mean(), line_dash="dash",
                      annotation_text="Overall Average")
        st.plotly_chart(fig, width='stretch')

        st.subheader("Song Duration Trend")
        duration_decade = df_filtered.groupby("decade")["duration_min"].mean().reset_index()
        fig = px.bar(duration_decade, x="decade", y="duration_min",
                     title="Average Duration by Decade",
                     labels={"duration_min": "Minutes"},
                     color="duration_min", color_continuous_scale="Viridis")
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Year information is not available in this dataset. Showing genre-level trends instead.")

        st.subheader("Genre Distribution")
        genre_counts = df["track_genre"].value_counts()
        fig = px.bar(genre_counts, x=genre_counts.index, y=genre_counts.values,
                     title="Track Count by Genre", color=genre_counts.values,
                     color_continuous_scale="Viridis", height=500)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, width='stretch')

        st.subheader("Audio Feature Comparison by Genre")
        feat_compare = st.selectbox("Feature", AUDIO_FEATURES, index=0)
        fig = px.box(df, x="track_genre", y=feat_compare,
                     title=f"{feat_compare.capitalize()} by Genre", height=500)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, width='stretch')

        st.subheader("Popularity Distribution")
        fig = px.histogram(df, x="popularity", nbins=30, height=500,
                           color_discrete_sequence=["coral"])
        st.plotly_chart(fig, width='stretch')

elif page == "Clusters":
    st.title("Cluster Analysis")

    st.subheader("Cluster Distribution")
    cluster_counts = df["cluster"].value_counts().sort_index()
    fig = px.bar(cluster_counts, x=cluster_counts.index, y=cluster_counts.values,
                 title="Songs per Cluster", color=cluster_counts.index,
                 color_continuous_scale="Viridis")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Cluster Profiles (Radar)")
    cluster_profile = df.groupby("cluster")[AUDIO_FEATURES].mean()
    fig = go.Figure()
    categories = AUDIO_FEATURES
    for c in cluster_profile.index:
        values = cluster_profile.loc[c].values.tolist() + [cluster_profile.loc[c].values[0]]
        fig.add_trace(go.Scatterpolar(r=values, theta=categories + [categories[0]],
                                      fill="toself", name=f"Cluster {c}"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      title="Audio Feature Profile by Cluster", height=500)
    st.plotly_chart(fig, width='stretch')

    st.subheader("PCA Visualization")
    fig = px.scatter(df.sample(min(10000, len(df)), random_state=42),
                     x="pca1", y="pca2",
                     color=df.sample(min(10000, len(df)), random_state=42)["cluster"].astype(str),
                     title="PCA (2D Projection)", opacity=0.5, height=600)
    st.plotly_chart(fig, width='stretch')

    st.subheader("Genre Composition by Cluster")
    cid = st.selectbox("Select Cluster", sorted(df["cluster"].unique()))
    cluster_genres = df[df["cluster"] == cid]["track_genre"].value_counts().head(10)
    fig = px.bar(cluster_genres, x=cluster_genres.index, y=cluster_genres.values,
                 title=f"Cluster {cid} - Top Genres", color=cluster_genres.values,
                 color_continuous_scale="Viridis")
    st.plotly_chart(fig, width='stretch')

elif page == "Recommender":
    st.title("Music Recommender")

    tab1, tab2, tab3 = st.tabs(["By Track Name", "By Audio Features", "By Genre"])

    with tab1:
        search_q = st.text_input("Search for a track name", "Dream")
        if search_q:
            matches = rec.search_tracks(search_q, n=10)
            if not matches.empty:
                track_choice = st.selectbox(
                    "Select a track",
                    matches.apply(lambda r: f"{r['track_name']} - {r['artists']} ({r['track_genre']})", axis=1)
                )
                if track_choice:
                    chosen_name = track_choice.split(" - ")[0]
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        genre_filter = st.selectbox("Genre Filter", ["All"] + sorted(df["track_genre"].unique()))
                    with col2:
                        n_recs = st.slider("Number of Recommendations", 5, 20, 10)

                    if st.button("Get Recommendations", key="btn_track"):
                        with st.spinner("Computing recommendations..."):
                            try:
                                results = rec.recommend_by_track(
                                    chosen_name, n=n_recs,
                                    genre_filter=genre_filter if genre_filter != "All" else None
                                )
                                st.dataframe(results, width='stretch')
                                fig = px.bar(results, x="score", y="track_name",
                                             orientation="h", color="score",
                                             color_continuous_scale="Viridis",
                                             title=f"Recommendations based on '{chosen_name}'")
                                st.plotly_chart(fig, width='stretch')
                            except ValueError as e:
                                st.error(str(e))
            else:
                st.info("No tracks found. Try a different search term.")

    with tab2:
        st.markdown("Adjust the audio feature sliders to find songs that match your preferences.")
        col1, col2 = st.columns(2)
        feature_input = {}
        for i, feat in enumerate(AUDIO_FEATURES):
            with col1 if i < len(AUDIO_FEATURES) // 2 else col2:
                feature_input[feat] = st.slider(feat.capitalize(), 0.0, 1.0,
                                                0.5 if feat in ["danceability"] else 0.3, 0.05)

        genre_filter2 = st.selectbox("Genre Filter (optional)", ["All"] + sorted(df["track_genre"].unique()), key="genre2")
        n_recs2 = st.slider("Number of Recommendations", 5, 20, 10, key="n2")

        if st.button("Get Recommendations", key="btn_feat"):
            with st.spinner("Computing recommendations..."):
                tempo_val = feature_input.get("tempo", 120)
                feature_input["tempo"] = tempo_val * 220
                feature_input["loudness"] = feature_input.get("loudness", 0.5) * -60 + 0.5
                feature_input["duration_ms"] = feature_input.get("duration_ms", 0.5) * 300000
                try:
                    results = rec.recommend_by_features(
                        feature_input, n=n_recs2,
                        genre_filter=genre_filter2 if genre_filter2 != "All" else None
                    )
                    st.dataframe(results, width='stretch')
                except Exception as e:
                    st.error(str(e))

    with tab3:
        genre3 = st.selectbox("Select Genre", sorted(df["track_genre"].unique()))
        min_pop = st.slider("Minimum Popularity", 0, 100, 50)
        n_recs3 = st.slider("Number of Songs", 5, 20, 10, key="n3")

        if st.button("Show Top Songs", key="btn_genre"):
            results = rec.recommend_by_genre(genre3, n=n_recs3, min_popularity=min_pop)
            if not results.empty:
                st.dataframe(results, width='stretch')
            else:
                st.info("No songs found with the given criteria.")

st.sidebar.markdown("---")
st.sidebar.info("Built with Python, Streamlit, Scikit-learn")
