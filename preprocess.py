"""
preprocess.py
NLP cleaning and vectorization functions for reviews.
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import numpy as np

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_review(text: str) -> str:
    """
    Clean review text: lowercase, remove punctuation/numbers, stopwords, lemmatize.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

def preprocess_reviews(df):
    """
    Apply cleaning to all reviews in a DataFrame column 'content'.
    """
    df = df.copy()
    df['cleaned'] = df['content'].astype(str).apply(clean_review)
    return df

def get_tfidf_matrix(texts, max_features=5000):
    """
    Fit and transform TF-IDF vectorizer on a list of texts.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf = vectorizer.fit_transform(texts)
    return tfidf, vectorizer
