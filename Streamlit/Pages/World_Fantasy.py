# ── Pages/world_fantasy.py ──────────────────────────────────────────────────
import json
import pickle
import base64
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import streamlit.components.v1 as components
import sys

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "Models"
ASSETS    = Path(__file__).resolve().parents[1] / "Assets"
sys.path.append(str(Path(__file__).resolve().parents[1]))
from Components.shared import set_page_style

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World Fantasy — The Other Shelf",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

set_page_style()

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    df           = pd.DataFrame(json.load(open(MODEL_DIR / "books_index.json", encoding="utf-8")))
    tfidf_matrix = sparse.load_npz(MODEL_DIR / "tfidf_matrix.npz")
    with open(MODEL_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return df, tfidf_matrix, vectorizer

df, tfidf_matrix, vectorizer = load_model()
print(f"✅ Loaded {len(df):,} books")

# ── Recommender functions ────────────────────────────────────────────────────
UNDERREPRESENTED = {
    "oceania", "australian-fantasy", "indigenous-fantasy", "indigenous_americas",
    "latin-american-fantasy", "latin_american", "south-american-fantasy",
    "middle-eastern-fantasy", "middle_eastern", "filipino", "southeast_asian",
    "african-science-fiction", "orisha", "igbo", "akan", "zulu", "yoruba",
    "anansi", "xianxia", "wuxia",
}

def similarity_label(score):
    if score >= 0.3:  return "Almost identical"
    if score >= 0.15: return "Very similar"
    if score >= 0.08: return "Similar"
    if score >= 0.04: return "Loosely related"
    return "Inspired by"

def build_text(row):
    parts = []
    if row.get("description"):
        parts.append(str(row["description"]))
    subjects = row.get("subjects", [])
    if isinstance(subjects, list) and subjects:
        parts.append(" ".join(str(s) for s in subjects))
    parts.append(str(row.get("title", "")) * 2)
    parts.append(str(row.get("author", "")) * 2)
    return " ".join(parts).strip()

def recommend_three_lanes(query, df, tfidf_matrix, vectorizer, 
                          search_by="title", top_n=5):
    if search_by == "title":
        matches = df[df["title"].str.contains(query, case=False, na=False)]
        if len(matches) == 0:
            return None, None, None, None
        idx        = matches.index[0]
        query_book = df.iloc[idx]
        query_vec  = tfidf_matrix[idx]
        query_author = query_book["author"].lower().strip()
    elif search_by == "author":
        matches = df[df["author"].str.contains(query, case=False, na=False)]
        if len(matches) == 0:
            return None, None, None, None
        idx        = matches.index[0]
        query_book = df.iloc[idx]
        query_vec  = tfidf_matrix[idx]
        query_author = query_book["author"].lower().strip()
    else:  # keywords
        query_vec    = vectorizer.transform([query])
        query_book   = None
        query_author = ""
        idx          = None

    sim_scores            = cosine_similarity(query_vec, tfidf_matrix).flatten()
    results               = df.copy()
    results["similarity"] = sim_scores
    if idx is not None:
        results = results.drop(index=idx)

    same_author = results[
        results["author"].str.lower().str.strip() == query_author
    ].sort_values("similarity", ascending=False).head(top_n) if query_author else pd.DataFrame()

    similar = results[
        results["author"].str.lower().str.strip() != query_author
    ].sort_values("similarity", ascending=False).head(top_n)

    hidden_pool = results[
        (results["source_tag"].isin(UNDERREPRESENTED)) &
        (results["author"].str.lower().str.strip() != query_author) &
        (results["similarity"] >= 0.02)
    ]
    hidden_gems = (
        hidden_pool
        .sort_values("similarity", ascending=False)
        .groupby("source_tag").first()
        .reset_index()
        .sort_values("similarity", ascending=False)
        .head(top_n)
    )

    return query_book, same_author, similar, hidden_gems

# ── Page header ──────────────────────────────────────────────────────────────
st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 0.5rem 0;'>
        <h1 style='font-size:2.8rem; color:#F5F0E8; 
                   text-shadow: 2px 2px 8px rgba(0,0,0,0.8);'>
            World Fantasy
        </h1>
        <p style='font-size:1rem; color:#D4C5A9;'>
            Fantasy & science fiction rooted in non-western mythologies and folklore
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Three columns ─────────────────────────────────────────────────────────────
left, middle, right = st.columns([1.2, 2, 1.2])

# ══ LEFT — About ══════════════════════════════════════════════════════════════
with left:
    st.markdown("""
        <div style='background:rgba(0,0,0,0.45); padding:1.5rem; 
                    border-radius:12px; border:1px solid rgba(255,255,255,0.15);'>
            <h3 style='color:#F5D78E;'>About this recommender</h3>
            <p style='color:#F5F0E8; line-height:1.7;'>
                Everyone reads the same 10 books.<br><br>
                This recommender helps you find something amazing from a different 
                heritage — fantasy rooted in African mythology, Japanese folklore, 
                Andean gods, Indigenous dreamtime, Arabian djinn, and much more.<br><br>
                <strong style='color:#F5D78E;'>3,995 books</strong> from traditions 
                beyond the western canon. Broaden your horizon.
            </p>
            <hr style='border-color:rgba(255,255,255,0.15);'>
            <p style='color:#D4C5A9; font-size:0.85rem;'>
                🌍 African & Diaspora<br>
                ⛩️ East & Southeast Asia<br>
                🕌 Middle East & Persia<br>
                🌿 Indigenous Americas<br>
                🌊 Oceania & Pacific<br>
                🌺 South Asia<br>
                🌎 Latin America
            </p>
        </div>
    """, unsafe_allow_html=True)

# ══ MIDDLE — Search ════════════════════════════════════════════════════════════
with middle:
    st.markdown("""
        <h3 style='color:#F5D78E; text-align:center;'>
            Find your next book
        </h3>
    """, unsafe_allow_html=True)

    search_mode = st.radio(
        "Search by:",
        ["Book title", "Author", "Keywords & themes"],
        horizontal=True,
        label_visibility="collapsed"
    )

    query = st.text_input(
        "Search",
        placeholder={
            "Book title":         "e.g. Children of Blood and Bone",
            "Author":             "e.g. Nnedi Okorafor",
            "Keywords & themes":  "e.g. japanese spirit world fox magic",
        }[search_mode],
        label_visibility="collapsed"
    )

    search_clicked = st.button("Search", use_container_width=True)

    if search_clicked and query:
        mode_map = {
            "Book title":        "title",
            "Author":            "author",
            "Keywords & themes": "keywords",
        }
        query_book, same_author, similar, hidden_gems = recommend_three_lanes(
            query, df, tfidf_matrix, vectorizer,
            search_by=mode_map[search_mode]
        )

        if query_book is None and mode_map[search_mode] != "keywords":
            st.warning(f"No results found for '{query}'. Try different keywords.")
        else:
            if query_book is not None:
                st.markdown(f"""
                    <div style='background:rgba(255,255,255,0.08); padding:1rem; 
                                border-radius:8px; margin-bottom:1rem;'>
                        <p style='color:#F5D78E; margin:0; font-size:0.85rem;'>
                            Showing results for:
                        </p>
                        <p style='color:#F5F0E8; margin:0; font-weight:bold;'>
                            {query_book['title']} — {query_book['author']}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            # ── Lane 1 ──────────────────────────────────────────────────────
            if len(same_author) > 0:
                st.markdown("<h4 style='color:#F5D78E;'>📚 More by this author</h4>",
                            unsafe_allow_html=True)
                for _, book in same_author.iterrows():
                    st.markdown(f"""
                        <div style='background:rgba(255,255,255,0.07); padding:0.8rem; 
                                    border-radius:8px; margin-bottom:0.5rem;
                                    border-left:3px solid #F5D78E;'>
                            <strong style='color:#F5F0E8;'>{book['title']}</strong><br>
                            <span style='color:#D4C5A9; font-size:0.85rem;'>{book['author']}</span>
                            <span style='float:right; color:#F5D78E; font-size:0.8rem;'>
                                ⭐ {book['avg_rating'] if pd.notna(book['avg_rating']) else 'N/A'}
                            </span>
                        </div>
                    """, unsafe_allow_html=True)

            # ── Lane 2 ──────────────────────────────────────────────────────
            st.markdown("<h4 style='color:#F5D78E;'>🌍 Similar books</h4>",
                        unsafe_allow_html=True)
            for _, book in similar.iterrows():
                label = similarity_label(book['similarity'])
                st.markdown(f"""
                    <div style='background:rgba(255,255,255,0.07); padding:0.8rem; 
                                border-radius:8px; margin-bottom:0.5rem;
                                border-left:3px solid #A8D5B5;'>
                        <strong style='color:#F5F0E8;'>{book['title']}</strong><br>
                        <span style='color:#D4C5A9; font-size:0.85rem;'>{book['author']}</span>
                        <span style='float:right; color:#A8D5B5; font-size:0.8rem;'>
                            {label}
                        </span><br>
                        <span style='color:#D4C5A9; font-size:0.8rem;'>
                            ⭐ {book['avg_rating'] if pd.notna(book['avg_rating']) else 'N/A'} 
                            · {int(book['num_ratings']):,} ratings
                        </span>
                    </div>
                """, unsafe_allow_html=True)

            # ── Lane 3 ──────────────────────────────────────────────────────
            if len(hidden_gems) > 0:
                st.markdown("<h4 style='color:#F5D78E;'>💎 Hidden gems from other heritages</h4>",
                            unsafe_allow_html=True)
                for _, book in hidden_gems.iterrows():
                    label = similarity_label(book['similarity'])
                    st.markdown(f"""
                        <div style='background:rgba(255,255,255,0.07); padding:0.8rem; 
                                    border-radius:8px; margin-bottom:0.5rem;
                                    border-left:3px solid #E8B4C0;'>
                            <strong style='color:#F5F0E8;'>{book['title']}</strong><br>
                            <span style='color:#D4C5A9; font-size:0.85rem;'>{book['author']}</span>
                            <span style='float:right; color:#E8B4C0; font-size:0.8rem;'>
                                {book['source_tag']} · {label}
                            </span><br>
                            <span style='color:#D4C5A9; font-size:0.8rem;'>
                                ⭐ {book['avg_rating'] if pd.notna(book['avg_rating']) else 'N/A'}
                                · {int(book['num_ratings']):,} ratings
                            </span>
                        </div>
                    """, unsafe_allow_html=True)

# ══ RIGHT — Random obscure books ══════════════════════════════════════════════
with right:
    st.markdown("""
        <h3 style='color:#F5D78E; text-align:center;'>
            Discover
        </h3>
        <p style='color:#D4C5A9; font-size:0.85rem; text-align:center;'>
            Obscure books from our collection
        </p>
    """, unsafe_allow_html=True)

    # Pick 5 random obscure books (low ratings, has cover)
    obscure = df[
        (df["num_ratings"] < 500) &
        (df["num_ratings"] > 0) &
        (df["cover_url"].str.startswith("http", na=False))
    ].sample(5, random_state=None)

    for _, book in obscure.iterrows():
        st.markdown(f"""
            <div style='background:rgba(0,0,0,0.35); padding:0.8rem; 
                        border-radius:8px; margin-bottom:0.8rem;
                        border:1px solid rgba(255,255,255,0.1);
                        text-align:center;'>
                <img src="{book['cover_url']}" 
                     style='height:120px; object-fit:contain; border-radius:4px;
                            margin-bottom:0.5rem;'
                     onerror="this.style.display='none'"/>
                <p style='color:#F5F0E8; font-size:0.8rem; margin:0; font-weight:bold;'>
                    {book['title'][:40]}{'...' if len(book['title']) > 40 else ''}
                </p>
                <p style='color:#D4C5A9; font-size:0.75rem; margin:0;'>
                    {book['author'][:30]}
                </p>
            </div>
        """, unsafe_allow_html=True)