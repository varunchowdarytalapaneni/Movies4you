"""
app.py
Flask app for movie recommendation system.
"""
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from tmdb_api import fetch_movie_metadata, fetch_movie_reviews, TMDBApiException
from preprocess import preprocess_reviews
from sentiment import SentimentAnalyzer
from recommend import build_content_matrix, hybrid_recommendation

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Movie Recommendation System</title>
    <style>
        body { font-family: Arial; margin: 40px; }
        .container { max-width: 600px; margin: auto; }
        input[type=text] { width: 80%; padding: 8px; }
        input[type=submit] { padding: 8px 16px; }
        .error { color: red; }
        ul { list-style: none; padding: 0; }
        li { margin-bottom: 10px; }
    </style>
</head>
<body>
<div class="container">
    <h2>Movie Recommendation System</h2>
    <form method="get" action="/">
        <input type="text" name="movie" placeholder="Enter a movie name" required>
        <input type="submit" value="Recommend">
    </form>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    {% if recommendations %}
    <h3>Top 5 Recommendations:</h3>
    <ul>
    {% for rec in recommendations %}
        <li><b>{{ rec.title }}</b> | Similarity: {{ rec.similarity }} | Positive Sentiment: {{ rec.positive_sentiment }}</li>
    {% endfor %}
    </ul>
    {% endif %}
</div>
</body>
</html>
'''

# Dummy movie list for demo; in production, fetch a set of movies from TMDB
MOVIE_TITLES = ["Inception", "The Matrix", "Interstellar", "The Dark Knight", "Fight Club", "Pulp Fiction", "Forrest Gump", "The Shawshank Redemption", "The Godfather", "The Lord of the Rings"]

@app.route("/recommend")
def recommend_api():
    movie = request.args.get("movie", "")
    if not movie:
        return jsonify({"error": "No movie provided."}), 400
    try:
        # Fetch metadata for all movies in MOVIE_TITLES
        movies = [fetch_movie_metadata(title) for title in MOVIE_TITLES]
        tfidf_matrix, _ = build_content_matrix(movies)
        # Fetch reviews for each movie and compute sentiment
        sentiment_scores = {}
        analyzer = SentimentAnalyzer()
        for m in movies:
            reviews_df = fetch_movie_reviews(m['id'])
            if not reviews_df.empty:
                reviews_df = preprocess_reviews(reviews_df)
                # For demo, label all reviews as positive (1)
                labels = [1]*len(reviews_df)
                analyzer.train(reviews_df['cleaned'], labels)
                pos_ratio = 1.0  # In real use, compute from predictions
            else:
                pos_ratio = 0.0
            sentiment_scores[m['title']] = pos_ratio
        recs = hybrid_recommendation(movie, movies, tfidf_matrix, sentiment_scores, top_n=5)
        return jsonify(recs)
    except TMDBApiException as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": "Internal server error."}), 500

@app.route("/", methods=["GET"])
def index():
    movie = request.args.get("movie", "")
    error = None
    recommendations = None
    if movie:
        try:
            movies = [fetch_movie_metadata(title) for title in MOVIE_TITLES]
            tfidf_matrix, _ = build_content_matrix(movies)
            sentiment_scores = {}
            analyzer = SentimentAnalyzer()
            for m in movies:
                reviews_df = fetch_movie_reviews(m['id'])
                if not reviews_df.empty:
                    reviews_df = preprocess_reviews(reviews_df)
                    labels = [1]*len(reviews_df)
                    analyzer.train(reviews_df['cleaned'], labels)
                    pos_ratio = 1.0
                else:
                    pos_ratio = 0.0
                sentiment_scores[m['title']] = pos_ratio
            recommendations = hybrid_recommendation(movie, movies, tfidf_matrix, sentiment_scores, top_n=5)
            if not recommendations:
                error = "Movie not found or not enough data."
        except TMDBApiException as e:
            error = str(e)
        except Exception as e:
            error = "Internal server error."
    return render_template_string(HTML_TEMPLATE, recommendations=recommendations, error=error)

if __name__ == "__main__":
    app.run(debug=True)
