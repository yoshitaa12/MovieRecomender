import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 Movie Recommendation System")

# Load dataset
movies = pd.read_csv("movies.csv")
movies['genres'] = movies['genres'].fillna('')
movies['combined'] = movies['title'] + ' ' + movies['genres']

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_movie(title, top_n=5):
    matches = movies[movies['title'].str.contains(title, case=False)]
    if matches.empty:
        return []
    idx = matches.index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return movies.iloc[[i[0] for i in scores]]['title']

# User input
movie = st.text_input("Enter a movie name")

if movie:
    recommendations = recommend_movie(movie)
    if len(recommendations) == 0:
        st.write("❌ Movie not found")
    else:
        st.write("### Recommended Movies:")
        for m in recommendations:
            st.write(m)
