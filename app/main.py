import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,600;1,9..40,300&display=swap');

/* ── Root Variables ── */
:root {
    --bg:        #0a0a0f;
    --surface:   #111118;
    --card:      #16161f;
    --border:    #2a2a3a;
    --accent:    #e8b84b;
    --accent2:   #c0392b;
    --text:      #e8e8f0;
    --muted:     #7a7a9a;
    --radius:    12px;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }
.block-container { padding: 0 2rem 4rem !important; max-width: 1400px !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0a0a0f 0%, #1a0a1f 40%, #0f1525 100%);
    border-bottom: 1px solid var(--border);
    padding: 3.5rem 2rem 2.5rem;
    margin: -1rem -2rem 3rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 80% at 80% 50%, rgba(232,184,75,0.07) 0%, transparent 60%),
                radial-gradient(ellipse 40% 60% at 10% 80%, rgba(192,57,43,0.06) 0%, transparent 50%);
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: clamp(3.5rem, 8vw, 7rem) !important;
    letter-spacing: 0.05em;
    line-height: 0.9;
    color: var(--text) !important;
    margin: 0 0 0.5rem !important;
    position: relative;
}
.hero-title span { color: var(--accent); }
.hero-sub {
    font-size: 1.05rem;
    color: var(--muted);
    font-weight: 300;
    letter-spacing: 0.02em;
    position: relative;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(232,184,75,0.1);
    border: 1px solid rgba(232,184,75,0.25);
    color: var(--accent);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 100px;
    margin-bottom: 1rem;
}
.hero-stats {
    display: flex; gap: 2rem; margin-top: 1.5rem; position: relative;
}
.stat { display: flex; flex-direction: column; }
.stat-num {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem; color: var(--accent); line-height: 1;
}
.stat-label { font-size: 0.72rem; color: var(--muted); letter-spacing: 0.08em; text-transform: uppercase; }

/* ── Section Headers ── */
.section-header {
    display: flex; align-items: center; gap: 12px;
    margin: 2.5rem 0 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
}
.section-icon {
    width: 36px; height: 36px;
    background: rgba(232,184,75,0.1);
    border: 1px solid rgba(232,184,75,0.2);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}
.section-title {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.6rem !important;
    letter-spacing: 0.06em;
    color: var(--text) !important;
    margin: 0 !important;
}
.section-desc { font-size: 0.8rem; color: var(--muted); margin-left: auto; }

/* ── Movie Card ── */
.movie-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.2rem;
    height: 100%;
    transition: border-color 0.2s, transform 0.2s;
    position: relative;
    overflow: hidden;
}
.movie-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    opacity: 0; transition: opacity 0.2s;
}
.movie-card:hover { border-color: rgba(232,184,75,0.3); transform: translateY(-2px); }
.movie-card:hover::before { opacity: 1; }
.card-rank {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.5rem;
    color: rgba(232,184,75,0.15);
    line-height: 1;
    margin-bottom: 0.3rem;
}
.card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.3;
    margin-bottom: 0.5rem;
}
.card-year {
    font-size: 0.75rem;
    color: var(--muted);
}
.card-genres {
    display: flex; flex-wrap: wrap; gap: 4px; margin-top: 0.6rem;
}
.genre-pill {
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--border);
    color: var(--muted);
    font-size: 0.65rem;
    padding: 2px 8px;
    border-radius: 100px;
    letter-spacing: 0.04em;
}
.card-rating {
    display: flex; align-items: center; gap: 4px;
    margin-top: 0.6rem;
}
.stars { color: var(--accent); font-size: 0.75rem; }
.rating-num { font-size: 0.8rem; font-weight: 600; color: var(--text); }
.rating-count { font-size: 0.7rem; color: var(--muted); }
.similarity-bar {
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    margin-top: 0.7rem;
    overflow: hidden;
}
.similarity-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    border-radius: 2px;
}
.similarity-label { font-size: 0.68rem; color: var(--muted); margin-top: 3px; }

/* ── Streamlit Widget Overrides ── */
.stSelectbox > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
}
.stSelectbox > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(232,184,75,0.15) !important;
}
.stSlider > div > div > div {
    background: var(--accent) !important;
}
.stButton > button {
    background: var(--accent) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 2rem !important;
    letter-spacing: 0.02em;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stMultiSelect > div > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: var(--radius) !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: var(--muted) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.2rem !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: var(--card) !important;
    color: var(--accent) !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

/* ── Alerts ── */
.result-header {
    background: rgba(232,184,75,0.07);
    border: 1px solid rgba(232,184,75,0.2);
    border-radius: var(--radius);
    padding: 0.9rem 1.2rem;
    margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 10px;
    font-size: 0.9rem; color: var(--accent);
    font-weight: 600;
}

/* ── Genre Grid ── */
.genre-btn-grid {
    display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1.5rem;
}
.genre-filter-btn {
    background: var(--card);
    border: 1px solid var(--border);
    color: var(--muted);
    padding: 5px 14px;
    border-radius: 100px;
    font-size: 0.78rem;
    cursor: pointer;
    transition: all 0.15s;
}
.genre-filter-btn.active, .genre-filter-btn:hover {
    background: rgba(232,184,75,0.1);
    border-color: var(--accent);
    color: var(--accent);
}

/* ── Dividers & misc ── */
hr { border-color: var(--border) !important; }
.stMetric { background: var(--card) !important; border-radius: var(--radius) !important;
            border: 1px solid var(--border) !important; padding: 0.8rem 1rem !important; }
.stMetric label { color: var(--muted) !important; font-size: 0.75rem !important; }
.stMetric [data-testid="metric-container"] > div { color: var(--accent) !important;
            font-family: 'Bebas Neue', sans-serif !important; font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ───────────────────────────────────────────────────────────────
GENRE_NAMES = [
    "unknown","Action","Adventure","Animation","Children's","Comedy",
    "Crime","Documentary","Drama","Fantasy","Film-Noir","Horror",
    "Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"
]

def _find_data_file(filename: str) -> str:
    from pathlib import Path

    here = Path(__file__).resolve()          # …/app/main.py
    repo_root = here.parent.parent           # …/movie-recommender/

    candidates = [
        repo_root / "ml-100k" / filename,   # repo_root/ml-100k/  ← Streamlit Cloud & local
        here.parent / "ml-100k" / filename, # app/ml-100k/
        repo_root / "data" / filename,
    ]

    # Brute-force: any ml-100k folder anywhere under repo root
    for p in repo_root.rglob(filename):
        candidates.append(p)

    for p in candidates:
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        f"'{filename}' not found. Repo root: {repo_root}. "
        f"Contents: {list(repo_root.iterdir())}"
    )


@st.cache_data
def load_data():
    item_path   = _find_data_file("u.item")
    rating_path = _find_data_file("u.data")

    # ── Movies ──
    cols = ["movie_id","title","release_date","video_release","imdb_url"] + GENRE_NAMES
    movies = pd.read_csv(item_path, sep="|", names=cols,
                         encoding="latin-1", index_col=False)

    # Parse year from title
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype("Int64")

    # Build genre list string for each movie
    def get_genres(row):
        return [g for g in GENRE_NAMES if row.get(g, 0) == 1]
    movies["genre_list"] = movies.apply(get_genres, axis=1)
    movies["genre_str"]  = movies["genre_list"].apply(lambda x: " ".join(x))

    # Combined feature for TF-IDF
    movies["features"] = movies["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True) \
                         + " " + movies["genre_str"] + " " + movies["genre_str"]  # double genre weight

    # ── Ratings ──
    ratings = pd.read_csv(rating_path, sep="\t",
                          names=["user_id","movie_id","rating","timestamp"])
    agg = ratings.groupby("movie_id").agg(
        avg_rating=("rating","mean"),
        num_ratings=("rating","count")
    ).reset_index()

    movies = movies.merge(agg, on="movie_id", how="left")
    movies["avg_rating"]  = movies["avg_rating"].fillna(0).round(2)
    movies["num_ratings"] = movies["num_ratings"].fillna(0).astype(int)

    # Weighted rating (Bayesian average)
    C = movies["avg_rating"].mean()
    m = movies["num_ratings"].quantile(0.60)
    movies["score"] = (movies["num_ratings"] / (movies["num_ratings"] + m)) * movies["avg_rating"] \
                     + (m / (movies["num_ratings"] + m)) * C

    return movies


@st.cache_data
def build_similarity(movies):
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    mat   = tfidf.fit_transform(movies["features"])
    sim   = cosine_similarity(mat)
    return sim


def star_string(rating, max_stars=5):
    filled  = int(round(rating / (10 / max_stars)))
    return "★" * filled + "☆" * (max_stars - filled)


def render_movie_card(row, rank=None, similarity=None):
    genres_html = "".join(
        f'<span class="genre-pill">{g}</span>' for g in row["genre_list"][:3]
    )
    year_str  = f"<span class='card-year'>{int(row['year'])}</span>" if pd.notna(row.get("year")) else ""
    rank_html = f"<div class='card-rank'>{rank:02d}</div>" if rank else ""
    stars     = star_string(row["avg_rating"] * 2) if row["avg_rating"] > 0 else ""

    sim_html = ""
    if similarity is not None:
        pct = int(similarity * 100)
        sim_html = f"""
        <div class='similarity-bar'>
            <div class='similarity-fill' style='width:{pct}%'></div>
        </div>
        <div class='similarity-label'>Match: {pct}%</div>"""

    rating_html = ""
    if row["avg_rating"] > 0:
        rating_html = f"""
        <div class='card-rating'>
            <span class='stars'>{stars}</span>
            <span class='rating-num'>{row['avg_rating']:.1f}</span>
            <span class='rating-count'>({row['num_ratings']:,} ratings)</span>
        </div>"""

    return f"""
    <div class='movie-card'>
        {rank_html}
        <div class='card-title'>{row['title']}</div>
        {year_str}
        <div class='card-genres'>{genres_html}</div>
        {rating_html}
        {sim_html}
    </div>"""


# ── Load ───────────────────────────────────────────────────────────────────────
movies = load_data()
sim_matrix = build_similarity(movies)
title_to_idx = pd.Series(movies.index, index=movies["title"]).to_dict()


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='hero'>
    <div class='hero-badge'>✦ ML-Powered Recommendations</div>
    <div class='hero-title'>CINE<span>MATCH</span></div>
    <div class='hero-sub'>TF-IDF content-based recommendations · MovieLens 100k dataset</div>
    <div class='hero-stats'>
        <div class='stat'>
            <span class='stat-num'>{len(movies):,}</span>
            <span class='stat-label'>Movies</span>
        </div>
        <div class='stat'>
            <span class='stat-num'>100K</span>
            <span class='stat-label'>Ratings</span>
        </div>
        <div class='stat'>
            <span class='stat-num'>{len(GENRE_NAMES)-1}</span>
            <span class='stat-label'>Genres</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🎯  Similar Movies",
    "🎭  Browse by Genre",
    "⭐  Top Rated",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Similar Movies
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class='section-header'>
        <div class='section-icon'>🎯</div>
        <span class='section-title'>Find Similar Movies</span>
        <span class='section-desc'>Based on title + genre similarity (TF-IDF + Cosine)</span>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c1:
        selected = st.selectbox(
            "Choose a movie",
            options=movies["title"].tolist(),
            index=0,
            label_visibility="collapsed",
            placeholder="Search for a movie...",
        )
    with c2:
        n_recs = st.slider("Results", 3, 20, 8, label_visibility="collapsed")

    _, btn_col, _ = st.columns([3, 1, 3])
    with btn_col:
        go = st.button("✦  Find Similar", use_container_width=True)

    if go and selected:
        idx = title_to_idx.get(selected)
        if idx is not None:
            scores = list(enumerate(sim_matrix[idx]))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            # skip self
            scores = [(i, s) for i, s in scores if i != idx][:n_recs]

            base_row = movies.iloc[idx]
            base_genres = " · ".join(base_row["genre_list"][:3]) if base_row["genre_list"] else "—"
            st.markdown(f"""
            <div class='result-header'>
                🎬 &nbsp;Because you liked <strong>{base_row['title']}</strong>
                &nbsp;·&nbsp; {base_genres}
            </div>""", unsafe_allow_html=True)

            cols = st.columns(4)
            for rank, (i, score) in enumerate(scores, 1):
                row = movies.iloc[i]
                with cols[(rank - 1) % 4]:
                    st.markdown(render_movie_card(row, rank=rank, similarity=score),
                                unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Browse by Genre
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class='section-header'>
        <div class='section-icon'>🎭</div>
        <span class='section-title'>Browse by Genre</span>
        <span class='section-desc'>Filter &amp; sort movies by genre · ranked by score</span>
    </div>""", unsafe_allow_html=True)

    BROWSABLE = [g for g in GENRE_NAMES if g != "unknown"]
    genre_cols = st.columns(5)
    selected_genres = []

    # Multi-select with pills aesthetic
    selected_genres = st.multiselect(
        "Select one or more genres",
        options=BROWSABLE,
        default=["Action"],
        label_visibility="visible",
    )

    g_sort = st.radio(
        "Sort by",
        ["⭐ Weighted Score", "🔢 Most Rated", "📅 Newest"],
        horizontal=True,
        label_visibility="collapsed",
    )
    g_count = st.slider("Number of movies to show", 4, 24, 12, key="genre_n")

    if selected_genres:
        mask = movies[selected_genres[0]] == 1
        for g in selected_genres[1:]:
            mask &= movies[g] == 1
        filtered = movies[mask].copy()

        if "Weighted Score" in g_sort:
            filtered = filtered.sort_values("score", ascending=False)
        elif "Most Rated" in g_sort:
            filtered = filtered.sort_values("num_ratings", ascending=False)
        else:
            filtered = filtered.sort_values("year", ascending=False)

        filtered = filtered.head(g_count)

        genre_label = " + ".join(selected_genres)
        st.markdown(f"""
        <div class='result-header'>
            🎭 &nbsp;Top <strong>{len(filtered)}</strong> {genre_label} films
        </div>""", unsafe_allow_html=True)

        cols = st.columns(4)
        for rank, (_, row) in enumerate(filtered.iterrows(), 1):
            with cols[(rank - 1) % 4]:
                st.markdown(render_movie_card(row, rank=rank), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("Select at least one genre above to start browsing.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Top Rated
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class='section-header'>
        <div class='section-icon'>⭐</div>
        <span class='section-title'>Top Rated Films</span>
        <span class='section-desc'>Bayesian weighted score · min 50 ratings</span>
    </div>""", unsafe_allow_html=True)

    decade_opts = ["All Decades", "1990s", "1980s", "1970s", "1960s", "Before 1960"]
    sort_opts   = ["⭐ Weighted Score", "🔢 Most Rated", "💯 Highest Avg Rating"]

    tc1, tc2, tc3 = st.columns([2, 2, 1])
    with tc1:
        decade = st.selectbox("Decade", decade_opts, label_visibility="visible")
    with tc2:
        t_sort = st.selectbox("Sort", sort_opts, label_visibility="visible")
    with tc3:
        t_count = st.slider("Show", 4, 24, 12, key="top_n")

    top = movies[movies["num_ratings"] >= 50].copy()

    decade_map = {
        "1990s": (1990, 1999), "1980s": (1980, 1989),
        "1970s": (1970, 1979), "1960s": (1960, 1969),
        "Before 1960": (0, 1959),
    }
    if decade in decade_map:
        lo, hi = decade_map[decade]
        top = top[(top["year"] >= lo) & (top["year"] <= hi)]

    if "Weighted Score" in t_sort:
        top = top.sort_values("score", ascending=False)
    elif "Most Rated" in t_sort:
        top = top.sort_values("num_ratings", ascending=False)
    else:
        top = top.sort_values("avg_rating", ascending=False)

    top = top.head(t_count)

    st.markdown(f"""
    <div class='result-header'>
        ⭐ &nbsp;Showing <strong>{len(top)}</strong> top-rated films
        {f' from the <strong>{decade}</strong>' if decade != "All Decades" else ""}
    </div>""", unsafe_allow_html=True)

    cols = st.columns(4)
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        with cols[(rank - 1) % 4]:
            st.markdown(render_movie_card(row, rank=rank), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

    # ── Genre breakdown chart ──
    st.markdown("""
    <div class='section-header' style='margin-top:2.5rem'>
        <div class='section-icon'>📊</div>
        <span class='section-title'>Dataset Overview</span>
    </div>""", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Movies",   f"{len(movies):,}")
    m2.metric("Total Ratings",  "100,000")
    m3.metric("Avg Rating",     f"{movies['avg_rating'].mean():.2f} / 5")
    m4.metric("Genres Covered", str(len(BROWSABLE)))

    st.markdown("<br>", unsafe_allow_html=True)
    genre_counts = {g: int(movies[g].sum()) for g in BROWSABLE}
    gc_df = pd.DataFrame({"Genre": list(genre_counts.keys()),
                          "Count": list(genre_counts.values())}).sort_values("Count", ascending=False)
    st.bar_chart(gc_df.set_index("Genre"), color="#e8b84b", height=280)
