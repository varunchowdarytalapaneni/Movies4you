"""
recommend.py
Content-based and hybrid recommendation logic.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def create_bag_of_words(movie):
    """
    Combine genres, overview, keywords, and cast into a single string.
    """
    genres = ' '.join([g['name'] for g in movie.get('genres', [])])
    overview = movie.get('overview', '')
    keywords = ' '.join([k['name'] for k in movie.get('keywords', {}).get('keywords', [])])
    cast = ' '.join([c['name'] for c in movie.get('credits', {}).get('cast', [])[:5]])
    return f"{genres} {overview} {keywords} {cast}"

def build_content_matrix(movies):
    """
    Build TF-IDF matrix for all movies' bag-of-words.
    """
    bags = [create_bag_of_words(m) for m in movies]
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf = vectorizer.fit_transform(bags)
    return tfidf, vectorizer

def get_similar_movies(movie_title, movies, tfidf_matrix, vectorizer, top_n=10):
    """
    Return top-N similar movies by cosine similarity.
    """
    idx = next((i for i, m in enumerate(movies) if m['title'].lower() == movie_title.lower()), None)
    if idx is None:
        return []
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i for i, score in sim_scores[1:top_n+1]]
    return [(movies[i]['title'], sim_scores[i][1]) for i in top_indices]

def hybrid_recommendation(movie_title, movies, tfidf_matrix, sentiment_scores, top_n=5):
    """
    Hybrid: combine similarity and sentiment for ranking.
    sentiment_scores: dict {title: positive_ratio}
    """
    idx = next((i for i, m in enumerate(movies) if m['title'].lower() == movie_title.lower()), None)
    if idx is None:
        return []
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    results = []
    for i, m in enumerate(movies):
        if i == idx:
            continue
        sim_score = cosine_sim[i]
        pos_sent = sentiment_scores.get(m['title'], 0)
        final_score = (sim_score * 0.7) + (pos_sent * 0.3)
        results.append({
            'title': m['title'],
            'similarity': round(sim_score, 3),
            'positive_sentiment': round(pos_sent, 3),
            'final_score': round(final_score, 3)
        })
    results = sorted(results, key=lambda x: x['final_score'], reverse=True)[:top_n]
    return results
