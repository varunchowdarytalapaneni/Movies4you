"""
tmdb_api.py
Functions for fetching movie metadata and reviews from TMDB API.
"""
import requests
import pandas as pd
from typing import List, Dict, Any

TMDB_API_KEY = "5a1ff503e759ebba1e06e628823"  # Replace with your TMDB API key
TMDB_BASE_URL = "https://api.themoviedb.org/3"

class TMDBApiException(Exception):
    pass

def fetch_movie_metadata(movie_title: str) -> Dict[str, Any]:
    """
    Fetch movie metadata from TMDB by title.
    Returns a dict with movie details or raises TMDBApiException.
    """
    try:
        params = {"api_key": TMDB_API_KEY, "query": movie_title}
        resp = requests.get(f"{TMDB_BASE_URL}/search/movie", params=params)
        resp.raise_for_status()
        data = resp.json()
        if not data["results"]:
            raise TMDBApiException("Movie not found.")
        movie_id = data["results"][0]["id"]
        details = requests.get(f"{TMDB_BASE_URL}/movie/{movie_id}", params={"api_key": TMDB_API_KEY, "append_to_response": "keywords,credits"})
        details.raise_for_status()
        return details.json()
    except requests.RequestException as e:
        raise TMDBApiException(f"API request failed: {e}")

def fetch_movie_reviews(movie_id: int) -> pd.DataFrame:
    """
    Fetch user reviews for a movie by TMDB movie ID.
    Returns a DataFrame with columns: [author, content].
    """
    try:
        params = {"api_key": TMDB_API_KEY}
        resp = requests.get(f"{TMDB_BASE_URL}/movie/{movie_id}/reviews", params=params)
        resp.raise_for_status()
        data = resp.json()
        reviews = data.get("results", [])
        df = pd.DataFrame(reviews)
        if not df.empty:
            df = df[["author", "content"]]
        return df
    except requests.RequestException as e:
        raise TMDBApiException(f"API request failed: {e}")
