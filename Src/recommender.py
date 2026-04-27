from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).parent.parent / "Data" / "non fiction" / "leftpolitics_final_clean.csv"


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def build_index(df: pd.DataFrame):
    corpus = (
        df["title"].fillna("") + " "
        + df["author"].fillna("") + " "
        + df["description"].fillna("")
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=25000)
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def recommend(query: str, df: pd.DataFrame, vectorizer, matrix, top_n: int = 10) -> pd.DataFrame:
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    top_idx = scores.argsort()[::-1][:top_n]
    results = df.iloc[top_idx].copy()
    results["score"] = scores[top_idx]
    return results[results["score"] > 0].reset_index(drop=True)
