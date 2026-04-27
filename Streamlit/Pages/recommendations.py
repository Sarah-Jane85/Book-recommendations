import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[2]))
from Src.recommender import build_index, load_data, recommend

st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")

st.title("📚 Book Recommender")
st.caption("Search by keyword, topic, author, or title — we'll find the closest matches in the collection.")


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_index(_df):
    return build_index(_df)


df = get_data()
vectorizer, matrix = get_index(df)

query = st.text_input(
    "What are you looking for?",
    placeholder="e.g. capitalism, colonialism, David Harvey, Capital...",
)
top_n = st.slider("Number of results", min_value=3, max_value=20, value=10)

if query.strip():
    results = recommend(query, df, vectorizer, matrix, top_n=top_n)

    if results.empty:
        st.info("No matches found. Try different keywords.")
    else:
        st.markdown(f"**{len(results)} results** for *{query}*")
        st.divider()

        for _, row in results.iterrows():
            year = f" · {int(row['year_published'])}" if pd.notna(row["year_published"]) else ""
            st.markdown(f"### {row['title']}")
            st.markdown(f"**{row['author']}**{year}")
            with st.expander("Description"):
                st.write(row["description"])
            st.divider()
