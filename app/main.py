import sys
from pathlib import Path

# Ensure project root is in path (important for Streamlit Cloud)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st
from app.utils import build_and_save_model, load_artifacts, recommend

# Page config

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Simple Movie Recommender — Demo")

# Load or build model

if not Path("models").exists() or not list(Path("models").glob("*")):
    with st.spinner("Building model (first run only)..."):
        movies = build_and_save_model()
else:
    movies, vectorizer, similarity = load_artifacts()

# UI Inputs

selected_movie = st.selectbox(
    "Choose a movie",
    sorted(movies["title"].tolist())
)

num_recs = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=20,
    value=5
)

# Recommendation Button

if st.button("Recommend"):
    recommendations = recommend(
        selected_movie,
        movies,
        similarity,
        top_n=num_recs
    )

    st.subheader("Top recommendations")

    for i, movie in enumerate(recommendations, start=1):
        st.write(f"{i}. 🎥 {movie}")
