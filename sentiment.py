"""
sentiment.py
Naive Bayes sentiment analysis for movie reviews.
"""
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        self.model = MultinomialNB()
        self.vectorizer = None

    def train(self, reviews, labels):
        X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        self.model.fit(X_train_vec, y_train)
        y_pred = self.model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        return acc

    def predict_sentiment(self, review_text):
        X_vec = self.vectorizer.transform([review_text])
        pred = self.model.predict(X_vec)[0]
        return 'Positive' if pred == 1 else 'Negative'
