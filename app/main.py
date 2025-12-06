import streamlit as st
from app.utils import build_and_save_model, load_artifacts, recommend
from pathlib import Path

st.set_page_config(page_title="Movie Recommender")

st.title("Simple Movie Recommender — Demo")

models_dir = Path(__file__).resolve().parents[1] / "models"
if not (models_dir / "similarity.npy").exists():
    st.info("Building model — this runs once. Please wait...")
    movies = build_and_save_model()

movies, tfidf, sim = load_artifacts()

movie_choice = st.selectbox("Choose a movie", movies['title'].tolist())
top_k = st.slider("Number of recommendations", min_value=3, max_value=20, value=5)

if st.button("Recommend"):
    recs = recommend(movie_choice, movies, sim, top_n=top_k)
    if not recs:
        st.write("No recommendations found.")
    else:
        st.write("Top recommendations:")
        for i, r in enumerate(recs, 1):
            st.write(f"{i}. {r}")
