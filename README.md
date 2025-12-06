🎬 Movie Recommender System

A simple content-based movie recommendation system built using:
Python
Streamlit
TF-IDF Vectorization
MovieLens 100k Dataset

This project recommends movies similar to a movie selected by the user using text features (title + genres) and cosine similarity.

📂 Project Structure
movie-recommender/
│
├── app/
│   ├── main.py          # Streamlit web application
│   └── utils.py         # Data loading, model building & recommendation functions
│
├── data/
│   └── movies.csv       # MovieLens dataset (converted from u.item)
│
├── models/
│   ├── tfidf.pkl        # Saved TF-IDF model
│   └── similarity.npy   # Precomputed similarity matrix (fast loading)
│
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation

📥 Dataset Used (MovieLens 100k)

Dataset Source: https://grouplens.org/datasets/movielens/100k/
The original file u.item was processed into a clean movies.csv with:

movie_id
title
genres

This CSV is used for building TF-IDF features.

🚀 Features

Recommend movies similar to any selected movie
Simple, fast, and lightweight
Runs completely in Python
Interactive web UI using Streamlit
Easy to deploy online

🛠️ How It Works (Behind the Scenes)

Load movies dataset
Create a description field = title + genres
Convert text into numerical features using TF-IDF Vectorizer
Compute cosine similarity matrix between all movies
When the user selects a movie → return the top N most similar movies

▶️ How to Run the Project
1. Install dependencies

    pip install -r requirements.txt

2. Run the Streamlit App

    streamlit run app/main.py

3. Open the browser

   Streamlit will automatically open your app or show a link like:

   http://localhost:8501

🌐 Deploying the App (Streamlit Cloud)

Push your code to GitHub
Go to https://share.streamlit.io
Click New App
Select your repo → choose main.py
Deploy
Your project gets a public shareable link instantly.


🔮 Future Improvements

Add posters, ratings & movie descriptions
Add collaborative filtering (user-based recommendations)
Add hybrid recommender (content + collaborative)
Add search bar & filters

Deploy on Render / Railway / GitHub Pages (stlite)

👨‍💻 Author

Your Name: Sanskar Kadam
Feel free to use, modify, and improve this project.