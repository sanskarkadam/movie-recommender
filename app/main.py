import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Page settings
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬"
)

st.title("🎬 Movie Recommender")
st.write("Find movies similar to your favorite movie.")


# Movie genres
GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
]


# Load movie data
@st.cache_data
def load_movies():

    # Find the movie file
    possible_files = [
        "ml-100k/u.item",
        "data/u.item"
    ]

    file_path = None

    for file in possible_files:
        if Path(file).exists():
            file_path = file
            break

    if file_path is None:
        st.error("Movie data file was not found.")
        st.stop()

    # Column names for MovieLens data
    columns = [
        "movie_id",
        "title",
        "release_date",
        "video_release",
        "imdb_url"
    ] + GENRES

    movies = pd.read_csv(
        file_path,
        sep="|",
        names=columns,
        encoding="latin-1"
    )

    # Get year from movie title
    movies["year"] = movies["title"].str.extract(
        r"\((\d{4})\)"
    )

    # Get genres for each movie
    def get_genres(row):

        movie_genres = []

        for genre in GENRES:
            if row[genre] == 1:
                movie_genres.append(genre)

        return movie_genres

    movies["genre_list"] = movies.apply(
        get_genres,
        axis=1
    )

    # Create text for recommendation system
    movies["features"] = (
        movies["title"] + " " +
        movies["genre_list"].apply(
            lambda x: " ".join(x)
        )
    )

    # Load ratings
    rating_file = "ml-100k/u.data"

    if Path(rating_file).exists():

        ratings = pd.read_csv(
            rating_file,
            sep="\t",
            names=[
                "user_id",
                "movie_id",
                "rating",
                "timestamp"
            ]
        )

        rating_data = ratings.groupby("movie_id").agg(
            average_rating=("rating", "mean"),
            number_of_ratings=("rating", "count")
        ).reset_index()

        movies = movies.merge(
            rating_data,
            on="movie_id",
            how="left"
        )

    else:
        movies["average_rating"] = 0
        movies["number_of_ratings"] = 0

    # Fill missing ratings
    movies["average_rating"] = movies[
        "average_rating"
    ].fillna(0)

    movies["number_of_ratings"] = movies[
        "number_of_ratings"
    ].fillna(0)

    return movies


# Create similarity matrix
@st.cache_resource
def create_similarity(movies):

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    movie_vectors = vectorizer.fit_transform(
        movies["features"]
    )

    similarity = cosine_similarity(
        movie_vectors
    )

    return similarity


# Display one movie
def show_movie(movie):

    st.subheader(movie["title"])

    if pd.notna(movie["year"]):
        st.write("Year:", movie["year"])

    st.write(
        "Genres:",
        ", ".join(movie["genre_list"])
    )

    if movie["average_rating"] > 0:
        st.write(
            "Rating:",
            round(movie["average_rating"], 2),
            "/ 5"
        )

    st.divider()


# Load data
movies = load_movies()

similarity = create_similarity(movies)


# Find movie index
movie_indexes = pd.Series(
    movies.index,
    index=movies["title"]
).to_dict()


# Create tabs
tab1, tab2, tab3 = st.tabs([
    "Similar Movies",
    "Browse by Genre",
    "Top Rated"
])


# --------------------------------------------------
# TAB 1: Similar Movies
# --------------------------------------------------

with tab1:

    st.header("Find Similar Movies")

    selected_movie = st.selectbox(
        "Choose a movie",
        movies["title"].tolist()
    )

    number_of_movies = st.slider(
        "Number of recommendations",
        3,
        10,
        5
    )

    if st.button("Recommend Movies"):

        movie_index = movie_indexes[selected_movie]

        # Get similarity scores
        scores = list(
            enumerate(similarity[movie_index])
        )

        # Sort by similarity
        scores = sorted(
            scores,
            key=lambda x: x[1],
            reverse=True
        )

        st.subheader("Recommended Movies")

        count = 0

        for index, score in scores:

            # Skip the selected movie
            if index == movie_index:
                continue

            movie = movies.iloc[index]

            st.write(
                f"**{movie['title']}**"
            )

            st.write(
                "Similarity:",
                round(score * 100, 2),
                "%"
            )

            st.write(
                "Genres:",
                ", ".join(movie["genre_list"])
            )

            st.divider()

            count += 1

            if count == number_of_movies:
                break


# --------------------------------------------------
# TAB 2: Browse by Genre
# --------------------------------------------------

with tab2:

    st.header("Browse Movies by Genre")

    selected_genre = st.selectbox(
        "Choose a genre",
        [genre for genre in GENRES if genre != "unknown"]
    )

    genre_movies = movies[
        movies[selected_genre] == 1
    ].copy()

    genre_movies = genre_movies.sort_values(
        "average_rating",
        ascending=False
    )

    st.write(
        "Movies in this genre:",
        len(genre_movies)
    )

    for _, movie in genre_movies.head(20).iterrows():

        show_movie(movie)


# --------------------------------------------------
# TAB 3: Top Rated Movies
# --------------------------------------------------

with tab3:

    st.header("Top Rated Movies")

    minimum_ratings = st.slider(
        "Minimum number of ratings",
        1,
        100,
        20
    )

    top_movies = movies[
        movies["number_of_ratings"] >= minimum_ratings
    ].copy()

    top_movies = top_movies.sort_values(
        "average_rating",
        ascending=False
    )

    for _, movie in top_movies.head(20).iterrows():

        show_movie(movie)