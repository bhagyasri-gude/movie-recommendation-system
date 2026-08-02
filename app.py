import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Title
st.title('🎬 Movie Recommendation System')

# Load Dataset
@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    movies = movies[['id', 'title', 'overview', 'genres']].dropna()
    return movies

movies = load_data()

# Preprocessing & Similarity calculation
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation Logic
def recommend(movie_title):
    try:
        idx = movies[movies['title'] == movie_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        movie_indices = [i[0] for i in sim_scores]
        return movies['title'].iloc[movie_indices].tolist()
    except:
        return []

# UI Dropdown
selected_movie = st.selectbox(
    'Type or select a movie from the dropdown:',
    movies['title'].values
)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    st.subheader('Recommended Movies for you:')
    for i, movie in enumerate(recommendations, 1):
        st.write(f"**{i}. {movie}**")