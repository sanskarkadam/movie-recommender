import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from pathlib import Path
import pickle

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)

def load_movies():
    """
    Robust loader for MovieLens 100k files.
    Tries (in order):
      1) data/movies.csv (standard CSV with columns movie_id, title, genres)
      2) data/u.item  (MovieLens ml-100k pipe-separated original file)
      3) data/ml-100k/u.item (if dataset is in a subfolder)
    Returns df with columns: movie_id (int), title (str), genres (comma string).
    """
    from pandas.errors import ParserError

    candidates = [
        DATA_DIR / "movies.csv",
        DATA_DIR / "u.item",
        BASE / "ml-100k" / "u.item",
        DATA_DIR / "ml-100k" / "u.item",
    ]

    candidates = [p for p in candidates if p.exists()]

    if not candidates:
        raise FileNotFoundError("No movie file found. Put movies.csv or the ml-100k folder in the data/ directory.")

    last_exc = None
    for path in candidates:
        try:
            df = pd.read_csv(path, encoding="latin-1")
            if 'title' in df.columns and ('genres' in df.columns or any(c.startswith('genre') for c in df.columns)):
                movies = df
            else:
                raise ParserError("force alternate parser")
            break
        except ParserError:
            try:
                cols = ["movie_id", "title", "release_date", "video_release", "imdb_url",
                        "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
                        "Crime", "Documentary", "Drama", "Fantasy", "FilmNoir", "Horror",
                        "Musical", "Mystery", "Romance", "SciFi", "Thriller", "War", "Western"]
                df2 = pd.read_csv(path, sep="|", encoding="latin-1", header=None, names=cols, engine="python")
                genre_cols = cols[5:]
                df2["genres"] = df2[genre_cols].apply(lambda row: ",".join([g for g, val in row.items() if val == 1]), axis=1)
                movies = df2[["movie_id", "title", "genres"]]
                break
            except Exception as e2:
                last_exc = e2
                continue
        except Exception as e:
            last_exc = e
            continue

    if 'movies' not in locals():
        raise RuntimeError(f"Failed to read movie file. Last error: {last_exc}")

    movies = movies.rename(columns={movies.columns[0]: "movie_id"}).copy()
    movies['movie_id'] = movies['movie_id'].astype(int)
    movies['title'] = movies['title'].astype(str).str.strip()
    if 'genres' not in movies.columns:
        movies['genres'] = ""
    movies['genres'] = movies['genres'].astype(str).str.replace('|', ',', regex=False)

    movies['description'] = (movies['title'].fillna('') + " " + movies['genres'].fillna(''))

    return movies.reset_index(drop=True)



def build_and_save_model():
    movies = load_movies()

    tfidf = TfidfVectorizer(stop_words='english', max_df=0.8)
    X = tfidf.fit_transform(movies['description'])

    sim = linear_kernel(X, X)

    with open(MODELS_DIR / "tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    np.save(MODELS_DIR / "similarity.npy", sim)

    return movies


def load_artifacts():
    movies = load_movies()

    with open(MODELS_DIR / "tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    sim = np.load(MODELS_DIR / "similarity.npy")

    return movies, tfidf, sim


def recommend(title, movies, sim_matrix, top_n=10):
    try:
        idx = movies.index[movies['title'] == title][0]
    except IndexError:
        return []

    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top = scores[1: top_n + 1]
    return [movies.iloc[i[0]].title for i in top]
