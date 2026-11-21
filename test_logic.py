"""Small checks for the dashboard's text-ranking approach."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def rank(profile, documents):
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform(documents + [profile])
    return cosine_similarity(matrix[-1], matrix[:-1]).flatten()


def test_relevant_topic_scores_higher():
    docs = ["machine learning model training", "accessible interface contrast"]
    scores = rank("machine learning", docs)
    assert scores[0] > scores[1]


def test_unrelated_profile_returns_zero_similarity():
    docs = ["machine learning model training", "accessible interface contrast"]
    scores = rank("classical music", docs)
    assert all(score == 0 for score in scores)
