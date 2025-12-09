🎬 Movie Recommender System

A clean, fast, and interactive content-based movie recommendation system built with Python + Streamlit + TF-IDF using the MovieLens dataset.

🚀 Live Demo

🔗 Streamlit Cloud App:
https://movie-recommender-sanskarkadam.streamlit.app/

📌 Project Overview

This is a lightweight content-based recommender system that suggests similar movies based on:

- Movie titles

- Genres

- TF-IDF text features

- Cosine similarity matrix

The app uses the MovieLens 100k dataset and provides instant recommendations via a clean Streamlit UI.
How It Works (Architecture)
Movie Dataset → Preprocessing → TF-IDF Vectorizer → Similarity Matrix → Streamlit UI → Recommendations

Breakdown:

Preprocess movies
Clean movie titles + convert genres into a unified text field.
Vectorization using TF-IDF
Converts text into numeric vectors.
Compute similarity matrix
Using cosine similarity on TF-IDF vectors.
Real-time recommendations
Select a movie → instantly view top similar titles.

-- Tech Stack --
Layer	Technology
Frontend UI	Streamlit
Backend Logic	Python
ML Technique	TF-IDF + Cosine Similarity
Dataset	MovieLens 100k
Deployment	Streamlit Cloud


-- Installation --

Clone the repo:
   git clone https://github.com/sanskarkadam/movie-recommender.git
   cd movie-recommender
Create virtual environment (Windows):
   python -m venv .venv
   .venv\Scripts\activate
Install dependencies:
  pip install -r requirements.txt
  ▶️ Run Locally
  streamlit run app/main.py
Open in browser:
  http://localhost:8501

-- Deploy on Streamlit Cloud --

-Push project to GitHub
-Go to https://share.streamlit.io
-Click New App
-Select:
-Repository: sanskarkadam/movie-recommender
-Branch: main
-File: app/main.py
-Deploy.

-- Features --

✔ Content-based similarity
✔ Streamlit UI
✔ Fast, lightweight model
✔ Clean architecture
✔ Fully deployable
✔ Beginner-friendly
✔ Expandable ML structure

-- Future Improvements --

Here are some upgrades you can add:

⭐ Poster images (TMDB API)

⭐ Actor/director metadata

⭐ Search bar with auto-suggest

⭐ Collaborative filtering (user-based recommendations)

⭐ Hybrid recommender (content + collaborative)

⭐ Advanced UI styling with custom components

I can help you implement any of these — just tell me!

-- Contributions --

Pull requests are welcome.
Found an issue? Open an issue on GitHub.

-- License --

MIT License — free to use, modify, and distribute.
